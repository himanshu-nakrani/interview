# AI Engineer (Generative AI) — Interview Preparation Guide

**19 parts · 87 sections · 4,009 questions · 2,041,651 words** — averaging 509 words per answer.

Written for a senior Python backend engineer moving into AI Engineer / Applied AI / GenAI Engineer roles at
the highest-paying employers in the world, weighted toward **AI Product / Applied** (Cursor, Perplexity,
Notion, Figma, Sierra, Harvey, Glean) and **Big-Tech / Enterprise Applied AI** (Meta, Google, Amazon,
Microsoft, Databricks, Stripe).

It assumes complete backend fluency — it will never explain a queue, an index, or a p99 — and assumes zero
prior ML, GPU, or training knowledge, teaching those from the ground up at full rigor.

[**Read the curriculum rationale, answer-format contract, and study paths →**](CURRICULUM.md)

## Read it on the web

The markdown in this repository is the source of truth. A small static-site generator renders it into a
reading site — one page per section, light/sepia/dark themes, adjustable type and measure, and search across
all 5,642 questions in both guides.

```bash
npm install
npm run dev      # build, then serve at http://localhost:4173
```

`npm run build` writes the site to `site/` (git-ignored). Two deployment paths are wired up, and they can be
used together or separately:

- **Vercel** — import the repository at [vercel.com/new](https://vercel.com/new). `vercel.json` sets the build
  command and output directory, so no settings need filling in. Every push builds a preview; `main` becomes
  production.
- **GitHub Pages** — `.github/workflows/pages.yml` builds and publishes on pushes to `main`. Requires
  **Settings → Pages → Source: GitHub Actions** once; creating the Pages site cannot be done from CI.

The build reads its base path from whichever host it runs on, so the same output works at a domain root or
under a `/repo/` sub-path.

Nothing in the generator writes to the markdown files. Section splits, question anchors, contents lists and
reading times are all derived at build time, so editing a `.md` file and rebuilding is the whole workflow.

## Every answer follows a five-beat contract

1. **Mental model first** — the intuition that makes the mechanism inevitable rather than memorized.
2. **Mechanism** — what actually happens, at the level of tensors, bytes, packets, or scheduler decisions.
3. **Code where code helps** — from-scratch implementations writable from memory; production snippets copy-paste correct.
4. **The trap** — the misconception or silent failure that passes review and breaks later.
5. **The production consequence** — milliseconds, dollars, or incidents, with the arithmetic shown.

Recurring markers: `⚠ Trap` · `🗣 Say this in the room` · `📐 Numbers you must know` · `📄 Paper` ·
`💰 Math` · `🔍 Failure taxonomy` · `🏋 Drill` · `📅 Volatile` (verify before your loop).

## Accuracy

Every section was independently fact-checked by an adversarial pass that recomputed each `💰 Math` and
`📐 Numbers` block, re-derived formulas, checked every `📄 Paper` attribution, and ran down code and API
claims. **758 corrections** were applied: 259 arithmetic errors, 152 false claims, 115 internal
contradictions, 61 unflagged stale facts, 46 code bugs, 44 formula errors, and 16 unverifiable citations.
150 of them were severe enough to fail a candidate who repeated them.

Numbers marked `📅 Volatile` — model names, prices, context limits, benchmark scores, regulatory dates —
were correct when written and should be re-verified before an interview loop. Section 5 is the single
refreshable place where the volatile landscape lives.

## Parts

| # | Part | Sections | Questions | Words |
|---|---|---|---|---|
| 0 | [Orientation, Routing and Getting the Interview](ai-engineering-guide/part-00-orientation-routing-and-getting-the-interview.md) | 6 | 237 | 121,066 |
| 1 | [Foundations Interviews Actually Probe](ai-engineering-guide/part-01-foundations-interviews-actually-probe.md) | 3 | 145 | 70,771 |
| 2 | [Model Internals](ai-engineering-guide/part-02-model-internals.md) | 8 | 353 | 170,036 |
| 3 | [Training and Post-Training](ai-engineering-guide/part-03-training-and-post-training.md) | 8 | 376 | 194,020 |
| 4 | [Inference, Serving and AI Infrastructure](ai-engineering-guide/part-04-inference-serving-and-ai-infrastructure.md) | 7 | 368 | 171,914 |
| 5 | [Retrieval and RAG](ai-engineering-guide/part-05-retrieval-and-rag.md) | 7 | 357 | 186,724 |
| 6 | [Agents and Tool Use](ai-engineering-guide/part-06-agents-and-tool-use.md) | 8 | 362 | 184,998 |
| 7 | [Context, Memory and Prompt Systems](ai-engineering-guide/part-07-context-memory-and-prompt-systems.md) | 4 | 180 | 85,801 |
| 8 | [Evaluation and Measurement (the spine)](ai-engineering-guide/part-08-evaluation-and-measurement-the-spine.md) | 5 | 232 | 117,143 |
| 9 | [The AI Product Backend, LLMOps and Reliability](ai-engineering-guide/part-09-the-ai-product-backend-llmops-and-reliability.md) | 4 | 205 | 100,713 |
| 10 | [Data and Enterprise Integration Platform](ai-engineering-guide/part-10-data-and-enterprise-integration-platform.md) | 2 | 88 | 49,924 |
| 11 | [Safety, Security and Governance](ai-engineering-guide/part-11-safety-security-and-governance.md) | 4 | 172 | 97,787 |
| 12 | [Beyond Text](ai-engineering-guide/part-12-beyond-text.md) | 4 | 170 | 86,768 |
| 13 | [Adjacent High-Comp Surfaces](ai-engineering-guide/part-13-adjacent-high-comp-surfaces.md) | 1 | 47 | 28,559 |
| 14 | [Coding Rounds](ai-engineering-guide/part-14-coding-rounds.md) | 4 | 200 | 102,364 |
| 15 | [System Design](ai-engineering-guide/part-15-system-design.md) | 4 | 187 | 107,474 |
| 16 | [Take-Homes, Work Trials and Defense](ai-engineering-guide/part-16-take-homes-work-trials-and-defense.md) | 4 | 153 | 81,055 |
| 17 | [Human Rounds](ai-engineering-guide/part-17-human-rounds.md) | 3 | 135 | 64,734 |
| 18 | [Offer, Leveling and Career](ai-engineering-guide/part-18-offer-leveling-and-career.md) | 1 | 42 | 19,800 |
| | **Total** | **87** | **4,009** | **2,041,651** |

## All 87 sections


**PART 0 — Orientation, Routing and Getting the Interview**  

- [1. Role Taxonomy, Loop Anatomy and the Company-by-Company Loop Map](ai-engineering-guide/part-00-orientation-routing-and-getting-the-interview.md#1-role-taxonomy-loop-anatomy-and-the-company-by-company-loop-map)
- [2. The Delta Map: What to Skip, What to Skim, What to Attack](ai-engineering-guide/part-00-orientation-routing-and-getting-the-interview.md#2-the-delta-map-what-to-skip-what-to-skim-what-to-attack)
- [3. Getting the Interview: Resume, GitHub, Portfolio and Positioning](ai-engineering-guide/part-00-orientation-routing-and-getting-the-interview.md#3-getting-the-interview-resume-github-portfolio-and-positioning)
- [4. Geography, Sponsorship, Remote and the India-to-Frontier-Lab Path](ai-engineering-guide/part-00-orientation-routing-and-getting-the-interview.md#4-geography-sponsorship-remote-and-the-india-to-frontier-lab-path)
- [5. The Model and Provider Landscape — the Volatility Sink](ai-engineering-guide/part-00-orientation-routing-and-getting-the-interview.md#5-the-model-and-provider-landscape-the-volatility-sink)
- [6. Navigating the Process: Take-Homes, Work Trials, Proctoring and Scope](ai-engineering-guide/part-00-orientation-routing-and-getting-the-interview.md#6-navigating-the-process-take-homes-work-trials-proctoring-and-scope)

**PART I — Foundations Interviews Actually Probe**  

- [7. Math for LLM Engineers: Linear Algebra, Probability, Information Theory, Statistics](ai-engineering-guide/part-01-foundations-interviews-actually-probe.md#7-math-for-llm-engineers-linear-algebra-probability-information-theory-statistics)
- [8. Training Math: Losses, Optimizers, Backprop, Numerics, Normalization, Stability](ai-engineering-guide/part-01-foundations-interviews-actually-probe.md#8-training-math-losses-optimizers-backprop-numerics-normalization-stability)
- [9. Classical ML, Pre-Transformer NLP, and the "Is an LLM Even Right Here?" Gate](ai-engineering-guide/part-01-foundations-interviews-actually-probe.md#9-classical-ml-pre-transformer-nlp-and-the-is-an-llm-even-right-here-gate)

**PART II — Model Internals**  

- [10. Attention From First Principles](ai-engineering-guide/part-02-model-internals.md#10-attention-from-first-principles)
- [11. KV-Cache-Aware Attention: MHA → MQA → GQA → MLA](ai-engineering-guide/part-02-model-internals.md#11-kv-cache-aware-attention-mha-mqa-gqa-mla)
- [12. Positional Encoding, Long Context, and SSM Hybrids](ai-engineering-guide/part-02-model-internals.md#12-positional-encoding-long-context-and-ssm-hybrids)
- [13. FFN, Activations, and Mixture-of-Experts (Serving-Weighted)](ai-engineering-guide/part-02-model-internals.md#13-ffn-activations-and-mixture-of-experts-serving-weighted)
- [14. Tokenization and the Token-Boundary Failure Class](ai-engineering-guide/part-02-model-internals.md#14-tokenization-and-the-token-boundary-failure-class)
- [15. Sampling and Decoding Algorithms](ai-engineering-guide/part-02-model-internals.md#15-sampling-and-decoding-algorithms)
- [16. Constrained Decoding and Structured Outputs](ai-engineering-guide/part-02-model-internals.md#16-constrained-decoding-and-structured-outputs)
- [17. Model Internals and Localization: Interpretability as a Debugging Skill](ai-engineering-guide/part-02-model-internals.md#17-model-internals-and-localization-interpretability-as-a-debugging-skill)

**PART III — Training and Post-Training**  

- [18. Pretraining, Data Curation, Scaling Laws and Mid-Training](ai-engineering-guide/part-03-training-and-post-training.md#18-pretraining-data-curation-scaling-laws-and-mid-training)
- [19. SFT, Chat Templates, Dataset Construction and Catastrophic Forgetting](ai-engineering-guide/part-03-training-and-post-training.md#19-sft-chat-templates-dataset-construction-and-catastrophic-forgetting)
- [20. Reward Models, Preference Data and RLHF with PPO](ai-engineering-guide/part-03-training-and-post-training.md#20-reward-models-preference-data-and-rlhf-with-ppo)
- [21. Direct Alignment: DPO, IPO, KTO, ORPO, SimPO](ai-engineering-guide/part-03-training-and-post-training.md#21-direct-alignment-dpo-ipo-kto-orpo-simpo)
- [22. RLVR, GRPO, Verifier Engineering and Reasoning-Model Training](ai-engineering-guide/part-03-training-and-post-training.md#22-rlvr-grpo-verifier-engineering-and-reasoning-model-training)
- [23. Distillation, Self-Improvement, Constitutional AI and Safety Training](ai-engineering-guide/part-03-training-and-post-training.md#23-distillation-self-improvement-constitutional-ai-and-safety-training)
- [24. PEFT: LoRA, QLoRA, Adapters, Merging and Multi-LoRA Serving](ai-engineering-guide/part-03-training-and-post-training.md#24-peft-lora-qlora-adapters-merging-and-multi-lora-serving)
- [25. The Escalation Ladder: Fine-Tune vs RAG vs Prompt vs Distill](ai-engineering-guide/part-03-training-and-post-training.md#25-the-escalation-ladder-fine-tune-vs-rag-vs-prompt-vs-distill)

**PART IV — Inference, Serving and AI Infrastructure**  

- [26. KV Cache Math, Memory Budgeting, Eviction and Offload](ai-engineering-guide/part-04-inference-serving-and-ai-infrastructure.md#26-kv-cache-math-memory-budgeting-eviction-and-offload)
- [27. Engine Internals and Kernels: PagedAttention, Continuous Batching, FlashAttention, Triton](ai-engineering-guide/part-04-inference-serving-and-ai-infrastructure.md#27-engine-internals-and-kernels-pagedattention-continuous-batching-flashattention-triton)
- [28. Prefill/Decode Disaggregation, Chunked Prefill and SLO Capacity Math](ai-engineering-guide/part-04-inference-serving-and-ai-infrastructure.md#28-prefilldecode-disaggregation-chunked-prefill-and-slo-capacity-math)
- [29. Prefix Caching, Prompt-Caching Economics and Speculative Decoding](ai-engineering-guide/part-04-inference-serving-and-ai-infrastructure.md#29-prefix-caching-prompt-caching-economics-and-speculative-decoding)
- [30. Quantization, Pruning, Sparsity and Model Compression](ai-engineering-guide/part-04-inference-serving-and-ai-infrastructure.md#30-quantization-pruning-sparsity-and-model-compression)
- [31. Inference Parallelism, Engine Selection, GPU Platform, Autoscaling and Batch Pipelines](ai-engineering-guide/part-04-inference-serving-and-ai-infrastructure.md#31-inference-parallelism-engine-selection-gpu-platform-autoscaling-and-batch-pipelines)
- [32. Distributed Training: FSDP/ZeRO, TP/PP/SP/CP/EP, Hardware and Cluster Reliability](ai-engineering-guide/part-04-inference-serving-and-ai-infrastructure.md#32-distributed-training-fsdpzero-tpppspcpep-hardware-and-cluster-reliability)

**PART V — Retrieval and RAG**  

- [33. Embedding Models: Training, Geometry, Selection and Domain Fine-Tuning](ai-engineering-guide/part-05-retrieval-and-rag.md#33-embedding-models-training-geometry-selection-and-domain-fine-tuning)
- [34. ANN Internals, Vector Databases, Filtering, Multi-Tenancy and Index Operations](ai-engineering-guide/part-05-retrieval-and-rag.md#34-ann-internals-vector-databases-filtering-multi-tenancy-and-index-operations)
- [35. Lexical Retrieval, Hybrid Fusion, Rerankers and Late Interaction](ai-engineering-guide/part-05-retrieval-and-rag.md#35-lexical-retrieval-hybrid-fusion-rerankers-and-late-interaction)
- [36. Chunking, Parsing, OCR and the Ingestion Pipeline](ai-engineering-guide/part-05-retrieval-and-rag.md#36-chunking-parsing-ocr-and-the-ingestion-pipeline)
- [37. RAG Architectures, Query Understanding and Long Context vs RAG](ai-engineering-guide/part-05-retrieval-and-rag.md#37-rag-architectures-query-understanding-and-long-context-vs-rag)
- [38. GraphRAG, Text-to-SQL and Multimodal RAG](ai-engineering-guide/part-05-retrieval-and-rag.md#38-graphrag-text-to-sql-and-multimodal-rag)
- [39. RAG Failure Taxonomy, Grounding, Citations and Retrieval Evaluation](ai-engineering-guide/part-05-retrieval-and-rag.md#39-rag-failure-taxonomy-grounding-citations-and-retrieval-evaluation)

**PART VI — Agents and Tool Use**  

- [40. Tool and Function Calling Mechanics](ai-engineering-guide/part-06-agents-and-tool-use.md#40-tool-and-function-calling-mechanics)
- [41. Agent Loops, Planning and Termination](ai-engineering-guide/part-06-agents-and-tool-use.md#41-agent-loops-planning-and-termination)
- [42. Agent Extension Mechanisms: Tools vs Skills vs MCP vs Code Execution](ai-engineering-guide/part-06-agents-and-tool-use.md#42-agent-extension-mechanisms-tools-vs-skills-vs-mcp-vs-code-execution)
- [43. MCP in Depth](ai-engineering-guide/part-06-agents-and-tool-use.md#43-mcp-in-depth)
- [44. Multi-Agent Topologies and Harness Engineering](ai-engineering-guide/part-06-agents-and-tool-use.md#44-multi-agent-topologies-and-harness-engineering)
- [45. Deep Research and Long-Horizon Agentic Search](ai-engineering-guide/part-06-agents-and-tool-use.md#45-deep-research-and-long-horizon-agentic-search)
- [46. Computer-Use, Browser and Coding Agents](ai-engineering-guide/part-06-agents-and-tool-use.md#46-computer-use-browser-and-coding-agents)
- [47. Durable Execution, Agent Reliability, Sandboxing, Permissions, Cost and Latency](ai-engineering-guide/part-06-agents-and-tool-use.md#47-durable-execution-agent-reliability-sandboxing-permissions-cost-and-latency)

**PART VII — Context, Memory and Prompt Systems**  

- [48. Context Engineering, Compaction and Context Rot](ai-engineering-guide/part-07-context-memory-and-prompt-systems.md#48-context-engineering-compaction-and-context-rot)
- [49. Agent Memory Architectures](ai-engineering-guide/part-07-context-memory-and-prompt-systems.md#49-agent-memory-architectures)
- [50. Prompt Engineering as Software and Automatic Optimization](ai-engineering-guide/part-07-context-memory-and-prompt-systems.md#50-prompt-engineering-as-software-and-automatic-optimization)
- [51. Reasoning Models, Thinking Budgets and Model Routing](ai-engineering-guide/part-07-context-memory-and-prompt-systems.md#51-reasoning-models-thinking-budgets-and-model-routing)

**PART VIII — Evaluation and Measurement (the spine)**  

- [52. Eval Strategy, Golden Sets, Error Analysis and Evaluation-Driven Development](ai-engineering-guide/part-08-evaluation-and-measurement-the-spine.md#52-eval-strategy-golden-sets-error-analysis-and-evaluation-driven-development)
- [53. LLM-as-Judge: Design, Bias and Calibration](ai-engineering-guide/part-08-evaluation-and-measurement-the-spine.md#53-llm-as-judge-design-bias-and-calibration)
- [54. Statistics, Benchmark Literacy and Harness Variance](ai-engineering-guide/part-08-evaluation-and-measurement-the-spine.md#54-statistics-benchmark-literacy-and-harness-variance)
- [55. Agent and Trajectory Evaluation](ai-engineering-guide/part-08-evaluation-and-measurement-the-spine.md#55-agent-and-trajectory-evaluation)
- [56. Online Evaluation, A/B Testing, Human-Eval Ops and Eval-Infrastructure Design](ai-engineering-guide/part-08-evaluation-and-measurement-the-spine.md#56-online-evaluation-ab-testing-human-eval-ops-and-eval-infrastructure-design)

**PART IX — The AI Product Backend, LLMOps and Reliability**  

- [57. Streaming APIs, Async Python, Queues and Idempotency Under Nondeterminism](ai-engineering-guide/part-09-the-ai-product-backend-llmops-and-reliability.md#57-streaming-apis-async-python-queues-and-idempotency-under-nondeterminism)
- [58. Gateways, Routing, Fallbacks, Caching Layers, Deployment and Model Migration](ai-engineering-guide/part-09-the-ai-product-backend-llmops-and-reliability.md#58-gateways-routing-fallbacks-caching-layers-deployment-and-model-migration)
- [59. Testing LLM Systems: Determinism, Fixtures and CI for a Nondeterministic Dependency](ai-engineering-guide/part-09-the-ai-product-backend-llmops-and-reliability.md#59-testing-llm-systems-determinism-fixtures-and-ci-for-a-nondeterministic-dependency)
- [60. Observability, Cost Engineering, On-Call and Incident Management](ai-engineering-guide/part-09-the-ai-product-backend-llmops-and-reliability.md#60-observability-cost-engineering-on-call-and-incident-management)

**PART X — Data and Enterprise Integration Platform**  

- [61. The AI Data Platform: Embedding Pipelines, Lakehouse Integration, Contracts and Versioning](ai-engineering-guide/part-10-data-and-enterprise-integration-platform.md#61-the-ai-data-platform-embedding-pipelines-lakehouse-integration-contracts-and-versioning)
- [62. The Enterprise Integration Surface: Connectors, Identity, Permission Mirroring and Tenant Configuration](ai-engineering-guide/part-10-data-and-enterprise-integration-platform.md#62-the-enterprise-integration-surface-connectors-identity-permission-mirroring-and-tenant-configuration)

**PART XI — Safety, Security and Governance**  

- [63. Prompt Injection and Agentic Attack Chains](ai-engineering-guide/part-11-safety-security-and-governance.md#63-prompt-injection-and-agentic-attack-chains)
- [64. Jailbreaks, Red Teaming, Guardrails and Content Safety](ai-engineering-guide/part-11-safety-security-and-governance.md#64-jailbreaks-red-teaming-guardrails-and-content-safety)
- [65. Privacy, Governance, Licensing, Supply Chain and Compliance](ai-engineering-guide/part-11-safety-security-and-governance.md#65-privacy-governance-licensing-supply-chain-and-compliance)
- [66. Frontier Safety Frameworks, Capability Evals and Model Specs](ai-engineering-guide/part-11-safety-security-and-governance.md#66-frontier-safety-frameworks-capability-evals-and-model-specs)

**PART XII — Beyond Text**  

- [67. Vision-Language Models and Document AI](ai-engineering-guide/part-12-beyond-text.md#67-vision-language-models-and-document-ai)
- [68. Realtime Voice and Speech Agents](ai-engineering-guide/part-12-beyond-text.md#68-realtime-voice-and-speech-agents)
- [69. Diffusion, Image, Video, 3D, Audio Generation and Diffusion LLMs](ai-engineering-guide/part-12-beyond-text.md#69-diffusion-image-video-3d-audio-generation-and-diffusion-llms)
- [70. Robotics, Vision-Language-Action Models, World Models and Physical AI](ai-engineering-guide/part-12-beyond-text.md#70-robotics-vision-language-action-models-world-models-and-physical-ai)

**PART XIII — Adjacent High-Comp Surfaces**  

- [71. Recsys and Search-Relevance Hybrids, On-Device, Small Models and Model-Choice Economics](ai-engineering-guide/part-13-adjacent-high-comp-surfaces.md#71-recsys-and-search-relevance-hybrids-on-device-small-models-and-model-choice-economics)

**PART XIV — Coding Rounds**  

- [72. From-Scratch Implementation: Model Internals and AI-Systems Primitives](ai-engineering-guide/part-14-coding-rounds.md#72-from-scratch-implementation-model-internals-and-ai-systems-primitives)
- [73. PyTorch/NumPy Fluency, OOM Triage and GPU Debugging](ai-engineering-guide/part-14-coding-rounds.md#73-pytorchnumpy-fluency-oom-triage-and-gpu-debugging)
- [74. Debug-the-Broken-Pipeline and Code-Review-the-Agent](ai-engineering-guide/part-14-coding-rounds.md#74-debug-the-broken-pipeline-and-code-review-the-agent)
- [75. AI-Assisted Coding Rounds, Pair Programming and DSA for AI Loops](ai-engineering-guide/part-14-coding-rounds.md#75-ai-assisted-coding-rounds-pair-programming-and-dsa-for-ai-loops)

**PART XV — System Design**  

- [76. The GenAI System-Design Framework and Tradeoff Vocabulary](ai-engineering-guide/part-15-system-design.md#76-the-genai-system-design-framework-and-tradeoff-vocabulary)
- [77. Case Catalog I — Retrieval, Documents, Search and Analytics Products](ai-engineering-guide/part-15-system-design.md#77-case-catalog-i-retrieval-documents-search-and-analytics-products)
- [78. Case Catalog II — Agents, Platform, Voice, Multimodal and AI Infrastructure](ai-engineering-guide/part-15-system-design.md#78-case-catalog-ii-agents-platform-voice-multimodal-and-ai-infrastructure)
- [79. AI Product Sense, Streaming UX and Designing for Model Failure](ai-engineering-guide/part-15-system-design.md#79-ai-product-sense-streaming-ux-and-designing-for-model-failure)

**PART XVI — Take-Homes, Work Trials and Defense**  

- [80. Canonical Take-Homes With Reference Solutions](ai-engineering-guide/part-16-take-homes-work-trials-and-defense.md#80-canonical-take-homes-with-reference-solutions)
- [81. The 48-Hour Paid Work Trial and Shipping Discipline](ai-engineering-guide/part-16-take-homes-work-trials-and-defense.md#81-the-48-hour-paid-work-trial-and-shipping-discipline)
- [82. The Defense, Walkthrough and "Show Me Something You Built" Rounds](ai-engineering-guide/part-16-take-homes-work-trials-and-defense.md#82-the-defense-walkthrough-and-show-me-something-you-built-rounds)
- [83. Technical Writing for AI Engineers](ai-engineering-guide/part-16-take-homes-work-trials-and-defense.md#83-technical-writing-for-ai-engineers)

**PART XVII — Human Rounds**  

- [84. Behavioral, Mission Alignment and Safety Reasoning](ai-engineering-guide/part-17-human-rounds.md#84-behavioral-mission-alignment-and-safety-reasoning)
- [85. Forward-Deployed Engineering: Decomposition Cases and Customer Simulation](ai-engineering-guide/part-17-human-rounds.md#85-forward-deployed-engineering-decomposition-cases-and-customer-simulation)
- [86. Research Literacy, the Paper Canon and Working With Research Teams](ai-engineering-guide/part-17-human-rounds.md#86-research-literacy-the-paper-canon-and-working-with-research-teams)

**PART XVIII — Offer, Leveling and Career**  

- [87. Leveling, Comp Mechanics, Offer Choreography and Closing](ai-engineering-guide/part-18-offer-leveling-and-career.md#87-leveling-comp-mechanics-offer-choreography-and-closing)

## Study paths

Every path assumes Part 0 (§1–§6) is mandatory — it is the routing and subtraction layer, and reading this guide linearly without it is the single most common way to run out of runway.

| Path | Weeks | Critical path (section numbers) |
|

## Companion guide

[`python_backend_interview_prep.md`](python_backend_interview_prep.md) — 1,633 questions on Python backend
engineering. This guide is the layer on top: it deliberately subtracts what that one already covers
(see §2, *The Delta Map*) and goes deep on the five rounds a backend engineer actually loses on —
transformer internals, post-training, serving internals, evaluation, and research literacy.
