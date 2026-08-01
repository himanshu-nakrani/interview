### Explain pre-filtering, post-filtering and filtered-HNSW, and tell me exactly where each one breaks.

Every real vector query has a predicate attached — tenant, workspace, ACL, date range, document type — and **how the engine combines the predicate with the ANN search is the single largest source of production retrieval bugs I have ever debugged.** There are three strategies and they have different failure regimes, so "which one does your engine use" is a question you must be able to answer about whatever you deploy.

**Post-filtering.** Run the ANN search ignoring the predicate, get top-K, then discard non-matching results. Simple, requires nothing of the index, and **catastrophic when the predicate is selective**. The ANN search does not know it should be looking among the matching subset, so it spends its whole budget in the wrong neighbourhood. This is the default in naive implementations and in every "just add a metadata filter" tutorial.

**Pre-filtering.** Resolve the predicate first using a normal index (a B-tree, an inverted index, a roaring bitmap over the payload), producing the exact set of matching IDs, then brute-force the distance over just those. Recall is **1.0 by construction** — it is flat search over a subset — and the cost is linear in the size of the matching set. Excellent when the set is small, unusable when it is large.

**Filtered graph traversal (filtered-HNSW).** Traverse the graph normally, but only admit nodes satisfying the predicate into the result heap, while still *routing through* non-matching nodes. This is the only approach that scales to both selective and unselective predicates — but the number of nodes you must visit grows roughly as `1/selectivity`, and worse, the graph's connectivity within the matching subset may be poor: if the matching nodes form islands connected only through non-matching nodes, greedy search may never reach some of them at all. That connectivity problem is what the research in this area is about.

**The decision procedure, which is what I actually want on the whiteboard:**

Let `s` be the selectivity (fraction of the corpus matching) and `N` the corpus size, so `|S| = s·N` is the matching-set size in *absolute* terms — and absolute size, not the ratio, is what decides.

- `|S| < ~50,000` → **pre-filter and brute force**. At d=1536, 50k vectors is `50,000 × 6144 = 307 MB`, a ~3 ms scan at 100 GB/s, and you get recall 1.0. Stop thinking about it. In practice this covers the majority of B2B multi-tenant workloads, which is a genuinely underappreciated fact.
- `|S| > ~50,000` and `s > ~0.1` → post-filtering with modest over-fetch is fine. At `s = 0.5`, over-fetching 2× gets you your k results with near-original recall.
- `|S| > ~50,000` and `s < ~0.1` → **you need filtered graph traversal.** Post-filter collapses; pre-filter is too slow.

**⚠ Trap:** the belief that a filter is "just a WHERE clause" that the vector store applies for free. In a relational database the planner picks between index paths based on estimated selectivity and you trust it. Most vector stores do not have a planner — they have one hard-coded strategy — and if that strategy is post-filtering, your recall silently depends on the selectivity of the filter, which varies per user, per tenant, and per query. **The same code path returns excellent results for your biggest customer and nothing for your smallest.** That asymmetry is exactly what makes it survive staging and QA.

**📄 Paper:** Gollapudi et al. (2023), *Filtered-DiskANN: Graph Algorithms for Approximate Nearest Neighbor Search with Filters* (WWW) — built label-aware graphs so that traversal restricted to a label's subset stays connected, rather than bolting a predicate onto a label-agnostic graph. Patel et al. (2024), *ACORN* (SIGMOD), took the predicate-agnostic route: expand each node's neighbourhood (effectively searching two hops) so the filtered subgraph stays navigable for arbitrary predicates without building a graph per predicate.

### Do the arithmetic for me. My filter matches 1% of the corpus and I post-filter with K=100. What happens?

This is the calculation I want a candidate to do live, because the numbers are shocking and they end the argument instantly.

Selectivity `s = 0.01`. You want `k = 10` results. You over-fetch `K = 100` from the ANN index, then apply the filter.

**📐 Numbers you must know:** the two formulas that decide every filtered-search argument are `E[survivors] = K · s` and `P[zero results] = (1 − s)^K`. Derive both from the binomial and you never need to memorise a table.

**Expected survivors** = `K × s = 100 × 0.01 = 1`. You asked for 10 and you expect 1.

**Probability of returning zero results** = `(1 − s)^K = 0.99^100`. Compute it: `ln(0.99) = −0.01005`, times 100 = `−1.005`, `e^(−1.005) = 0.366`. **36.6% of your queries return literally nothing.** The user's question hits your RAG pipeline, retrieval returns an empty list, and the LLM — being helpful — answers from parametric memory and hallucinates. No error is logged. Your p99 latency looks great.

**What K do you need?** For 10 expected survivors you need `K = 1,000`. But expectation is not enough; you want ≥10 with high probability. The count is Binomial(K, 0.01) with mean `0.01K` and standard deviation `sqrt(0.01K × 0.99) ≈ sqrt(0.01K)`. For a 95% one-sided guarantee you need `0.01K − 1.645·sqrt(0.01K) ≥ 10`. Let `μ = 0.01K`: `μ − 1.645√μ ≥ 10` → `√μ ≈ 4.06` → `μ ≈ 16.5` → **`K ≈ 1,650`**.

**What does K=1,650 cost?** `efSearch` must be ≥ K, so you go from `ef=100` to `ef=1650`, a 16.5× increase in nodes visited. HNSW latency is roughly linear in `ef` in this regime, so a 1.2 ms query becomes **~20 ms**. And your recall relative to *filtered ground truth* is still not 1.0, because the top-1650 unfiltered neighbours are not the top-10 filtered neighbours — you have only made it likely that some matching items appear, not that the *best* matching items do.

Now push it. `s = 0.001` (a mid-sized tenant in a 10,000-tenant system): you need `K ≈ 16,500`, `efSearch = 16,500`, roughly 200 ms per query, and at that point you are visiting 16,500 of your N nodes — if N is 10M you are scanning 0.17% of the corpus through a graph, which is *slower* than pre-filtering and brute-forcing the 10,000 matching vectors (`10,000 × 6144 = 61 MB`, **0.6 ms**).

**🗣 Say this in the room:** "Post-filtering at 1% selectivity with an over-fetch of 100 returns nothing at all on 37% of queries — that's `0.99^100`. To get ten results with 95% confidence you need to over-fetch about 1,650, which is a 16× latency hit and still doesn't guarantee you found the *best* matching neighbours. Below roughly 10% selectivity, post-filtering isn't a tuning problem, it's a wrong architecture, and I'd either pre-filter into a brute-force scan or move to an engine with native filtered traversal."

**🔍 Failure taxonomy — how post-filter collapse presents in production:** users report "it says it doesn't know about things I know are in there," and the reports cluster on *small tenants and narrow filters*. Retrieval returns 0–3 chunks instead of 10. The tell in your telemetry is the **distribution of `len(retrieved_chunks)`** — if you are not logging that, log it today. A healthy system returns exactly k almost always; a collapsing one has a long left tail. It is the cheapest possible detector for the most expensive possible retrieval bug.

### How does Qdrant's filterable HNSW actually work, and why is cardinality estimation the crux?

The crux is that **there is no single right strategy, so the engine must choose per query — which means it needs a cardinality estimate, which means it needs statistics, which is exactly the problem a relational query planner solves.** Qdrant is the clearest open-source implementation of that idea in a vector store, which is why it is worth understanding specifically.

Three pieces:

**Payload indexes.** You explicitly index the metadata fields you filter on — keyword, integer, geo, full-text, and a special `is_tenant`-style configuration for the tenancy field. These are ordinary inverted structures mapping a value to a set of point IDs, and they let the engine both *evaluate* the filter and *estimate its cardinality* without scanning.

**The `full_scan_threshold` decision.** Given a filter, the engine estimates how many points match. If that estimate is below a configured threshold (order of 10,000 points by default), it **pre-filters and brute-forces**: retrieve the matching IDs from the payload index, compute exact distances, done — recall 1.0, and fast because the set is small. This is exactly the `|S| < 50,000` rule from earlier, implemented as an automatic decision rather than something you have to reason about per query. That single feature eliminates the majority of filtered-search pathology in multi-tenant workloads.

**Filtered graph traversal above the threshold.** When the matching set is large, it searches the HNSW graph with the filter applied as an admission condition. The connectivity problem is real here, and Qdrant's answer is to build **additional links during indexing that account for the filters you actually use** — if you tell it that `tenant_id` is a tenancy field, it can structure the graph so that points sharing a tenant remain mutually navigable rather than relying on paths through other tenants' points. The generalisation of this idea is what Filtered-DiskANN and ACORN attack from different directions.

**Segment-based storage** underneath all of it: writes go to new segments, an optimizer merges them in the background, and deletes are reclaimed by that merge. That gives you the compaction story that a monolithic HNSW does not have.

**⚠ Trap:** filtering on a field you never added a payload index for. The filter still *works* — results are correct — but the engine must scan to evaluate it and cannot estimate its cardinality, so it falls back to the wrong strategy. The symptom is a query that is 100× slower than the identical query on an indexed field, with no error and no warning. In review I treat "every field appearing in a filter has a declared payload index" as a hard checklist item, exactly the same way I treat "every column in a WHERE clause on a hot path has an index" in Postgres. It is the same discipline; only the failure mode is unfamiliar.

### Design ACL-aware retrieval for an enterprise search product. Assume documents come from Drive, Slack, Confluence and Jira, each with its own permission model.

This is a Glean-shaped problem and it is one of the highest-signal design questions in this space, because the naive answer is one sentence and the correct answer has five hard parts.

**The naive answer:** denormalise a `visible_to` array of principal IDs onto every chunk, expand the querying user into their principal set, and filter `visible_to && $principals`. That is the right *skeleton*. Now the hard parts.

**Hard part 1 — the asymmetry of errors.** A false allow leaks a document; a false deny annoys a user. These are not comparable. So every design decision breaks toward denial, and the freshness SLA must be **asymmetric**: revocations must propagate faster than grants. In practice that means the CDC path for permission changes is separate from and higher-priority than the content path, and a revocation event is applied to the filter layer *before* it is applied to the index — I want a fast deny-list that is consulted at query time even when the index has not caught up.

**Hard part 2 — group expansion.** Permissions are granted to groups, groups nest, and a user in a large org can resolve to thousands of principals. Two directions to denormalise: expand groups into users at *write* time (so `visible_to` is a list of user IDs — fast to filter, but a single group-membership change rewrites every document that group can see, which can be millions), or expand the user into groups at *read* time (so `visible_to` holds group IDs and the query carries the user's principal set — cheap writes, but now the query filter is an `IN` over potentially 10,000 terms). **Read-time expansion is almost always correct**, because group membership changes far more often than you can afford to rewrite indexes, and because it is the version that gets revocation right instantly.

**Hard part 3 — the filter is now enormous and selective, which is the worst case from the previous questions.** A 10,000-term `IN` with 0.5% selectivity is precisely the post-filter-collapse regime. The fix is a **two-level filter**: index a coarse, low-cardinality *container* key on every chunk — the Drive folder, the Slack channel, the Confluence space, the Jira project — and filter on the user's accessible containers (typically tens to hundreds, not thousands) to get a candidate set, then apply the exact, expensive principal check on the small survivor set. You have converted one brutal filter into a cheap selective one plus a cheap exact one.

**Hard part 4 — deny rules and sharing semantics.** Real systems have explicit denies, "anyone with the link," "anyone in the org," inherited-versus-broken inheritance, and per-field restrictions. A pure allow-list cannot express a deny that overrides an inherited allow. My rule: **model the effective permission as a computed allow-set plus an explicit deny-set, resolve them at ingest into a boolean, and store the resolution's inputs so you can recompute and audit.** If you cannot recompute why a document was visible, you cannot pass a security review.

**Hard part 5 — the late-binding check.** Even with all of the above, the index is eventually consistent with the source systems. So the last stage before results reach the model is a **verification pass against the live permission service** for the (small) set of chunks you are about to use. Over-fetch by ~2× so that dropping a few late-denied chunks does not leave you short. This costs one batched RPC per query — say 15 ms — and it is the difference between "we filter on our index" and "we enforce authorisation." Only the second one survives an enterprise security review.

**🗣 Say this in the room:** "I'd denormalise principals onto chunks and expand the *user* into groups at query time, not the reverse, because group membership churns far faster than I can rewrite an index. To avoid the selective-filter collapse I'd filter first on a low-cardinality container key — folder, channel, space — then do the exact principal check on survivors, then do a live late-binding permission check on the final ~20 chunks before they reach the model. And I'd treat revocation as a separate, higher-priority path than content updates, because a false allow is an incident and a false deny is a ticket."

### A user resolves to ten thousand group IDs. Now what?

This is the concrete version of hard part 3 and it deserves its own treatment because the naive implementations all fail differently.

**What breaks.** A filter of the form `visible_to && ARRAY[10,000 ids]` means: for every candidate the engine considers, evaluate a 10,000-way set intersection. If the engine evaluates that predicate per-candidate during graph traversal, you have made every distance computation 10× more expensive. If it evaluates it as an inverted-index lookup, it must union 10,000 posting lists to build the matching set, which at even 10,000 docs per group is a 100M-element union.

**Fix 1 — roaring bitmaps and precomputed unions.** Represent each principal's accessible-document set as a roaring bitmap. The user's accessible set is the union of their principals' bitmaps. Unioning 10,000 roaring bitmaps is fast (they are compressed and union is the cheap operation), and — critically — you **cache the result per user with a short TTL and an invalidation hook on permission change**. Now the per-query cost is one bitmap fetch and one membership test per candidate, which is O(1)-ish. The memory cost: for 100M documents, a dense bitmap is 12.5 MB uncompressed and roaring will typically compress a realistic access set to a small fraction of that; a cache of 10,000 active users' bitmaps at ~1 MB each is 10 GB, which is a real but affordable Redis-shaped problem.

**Fix 2 — reduce cardinality at the source.** Ten thousand groups usually means the identity system has nested groups that have never been rationalised, plus a lot of groups that grant access to nothing in your corpus. Intersect the user's principal set with the set of principals that actually appear in the index: if only 340 of their 10,000 groups grant access to any indexed document, your filter is 340 terms. This is a cheap precomputation and it routinely removes an order of magnitude.

**Fix 3 — the container-key indirection from the previous answer.** Filter on ~50 container IDs instead of 10,000 principal IDs, then exact-check.

**Fix 4 — accept the pre-filter.** If the user's accessible set is, say, 80,000 documents out of 100M, then `s = 0.0008` — and the *absolute* size is 80,000, which is a 490 MB brute-force scan, ~5 ms. Recall 1.0. **For a large fraction of enterprise users, the correct filtered-search strategy is no ANN index at all.** I would rather do a 5 ms exact scan over a bitmap-selected subset than a 200 ms filtered graph traversal that gives me 0.87 recall, and I have made exactly that call in production.

**⚠ Trap:** caching the expanded principal set without an invalidation path. You have now built a system where firing someone does not revoke their search access for up to your TTL. Every permission cache needs (a) a short TTL, (b) an explicit invalidation on the revocation event, and (c) the late-binding check that makes a stale cache a performance issue rather than a security one. If you only get one of the three, take the late-binding check.

### Namespace-per-tenant or filter-by-tenant? Make the call and defend it.

The decision rule is short and I will give it first: **namespace (or physical partition) per tenant when tenant count is in the thousands or below and tenants are large; filter-by-tenant when tenant count is in the tens of thousands or above and the distribution is long-tailed — and then push hard for the engine to be able to pre-filter, because a tenant filter is exactly the selective filter that kills post-filtering.**

**Namespace-per-tenant** means each tenant gets its own physical index. What you get:

- **Isolation is structural, not logical.** A bug in your filter code cannot leak across tenants because there is no shared structure to leak through. That is a security property you can explain to a customer's security team in one sentence, and it is worth a lot in enterprise sales.
- **Search cost scales with tenant size, not corpus size.** A tenant with 2,000 chunks does a 12 MB scan. In a shared index they would pay to traverse a graph built over 100M vectors.
- **Per-tenant operations become possible**: delete a tenant (drop the index — GDPR erasure in one operation instead of a scan-and-delete), reindex a tenant, restore a tenant, move a tenant to different hardware, offload an inactive tenant to cold storage.
- **Noisy-neighbour containment.** One tenant's ingest burst rebuilds one tenant's index.

What it costs:

- **Fixed overhead per namespace.** Every index has metadata, file handles, memory arenas, and often a minimum allocation. At 100,000 tenants averaging 500 vectors, per-namespace overhead can dwarf the data itself. This is the reason the rule has a tenant-count bound.
- **Poor packing.** Small tenants leave RAM stranded.
- **Cross-tenant queries become fan-out.** If you ever need "search across all tenants" — for internal analytics, or for a shared knowledge base — you now have a scatter-gather.

**Filter-by-tenant** inverts all of that: perfect packing, one index to operate, trivial cross-tenant queries, and **isolation that is only as good as your filter code, on every single query path**.

The synthesis I actually deploy at scale is **hybrid tiering**: large tenants (top few hundred by volume) get their own namespace; the long tail shares a pooled index with a mandatory tenant filter and a pre-filter strategy; and there is a promotion path that moves a tenant into its own namespace when it crosses a size threshold. This is the same pattern as dedicated-versus-shared database schemas in any B2B SaaS, and you already know how to run it — the only new part is that the "shared" case has a recall cliff attached to it that a relational shared schema does not.

**⚠ Trap:** implementing the tenant filter in application code rather than making it structurally impossible to omit. The rule I enforce: **the tenant ID is never a parameter the caller passes; it is injected by a single retrieval client from the authenticated request context, and the raw index client is not exported.** Same discipline as row-level security. A code path that *can* query without a tenant filter will eventually be called without one — usually by a background job, an admin tool, or an eval harness that someone pointed at production.

### Describe the noisy-neighbour failure in a shared vector index. How does it present and how do you fix it?

In a shared index, tenants contend for four distinct resources, and each produces a different symptom — so the first diagnostic step is figuring out *which* resource is contended.

**Query CPU.** One tenant runs an agent that issues forty retrieval calls per user turn, at `efSearch=400`, across a corpus where their filter matches 40% of the index. Their queries are 50× more expensive than everyone else's. Symptom: global p99 rises, and it rises for *everyone*, including tenants doing nothing. This is the classic case and the fix is the one you already know — **per-tenant token-bucket rate limiting, but on cost units rather than request count**. The cost unit for a vector query is roughly `efSearch × M` distance computations plus the filter evaluation; charge a request's bucket proportional to that, not 1 per query. A tenant issuing 10 QPS at `ef=1000` should be throttled before a tenant issuing 100 QPS at `ef=50`.

**Memory / cache.** A large tenant's vectors are hot and evict everyone else's from the page cache. Symptom: small tenants see bimodal latency — fast when warm, 50× slower on a cold read — while the big tenant is uniformly fast. This one is invisible on a global dashboard because the big tenant dominates the query count and therefore dominates your p50 and even your p95. **You must slice retrieval latency by tenant.** The fix is either capacity (more RAM), or tiering (give the whale its own node), or namespace promotion.

**Write / compaction.** One tenant bulk-loads 5M documents. The index's background optimizer spends the next six hours merging segments, consuming CPU and IO, and every tenant's queries slow down. Symptom: a global latency step-change that correlates with nothing in the query logs and everything in the ingest logs. Fixes: throttle ingest per tenant (a token bucket on write bytes), schedule bulk loads into a separate index and swap, or physically separate the write path from the read replicas.

**Index structure itself — the subtle one.** In a shared HNSW, the graph is built over *all* tenants' vectors. A tenant whose documents are semantically unusual — a different language, a different domain — creates a region of the graph that other tenants' queries never visit, but which still inflates `N` and therefore the expected path length for everyone. More insidiously, if tenant A uploads 90% of the corpus, the graph's global structure is optimised for A's distribution, and tenant B's filtered traversal has to route through A's territory to get anywhere. **This is a recall problem, not just a latency problem, and it does not exist in the namespace-per-tenant design.**

**💰 Math — the whale argument.** Suppose one tenant is 60% of your 100M-vector index and 5% of your revenue. Their vectors cost `60e6 × 6274 bytes = 376 GB` of resident memory, roughly `376 GB × $4/GB-month = $1,504/month` — plus they are degrading every other tenant's latency. Moving them to a dedicated node costs the same $1,504 in isolation but recovers the shared tier's performance and, more usefully, makes the cost *attributable*. **Per-tenant cost attribution is the prerequisite for every pricing and capacity decision you will make**, and a shared index without per-tenant metrics makes it impossible. Emit vectors-stored, queries, and distance-computations per tenant from day one.

### Name every path by which data leaks across tenants in a retrieval system. Be exhaustive.

I would answer this as a checklist, because exhaustiveness is the point and because I have seen most of these actually happen.

**1. The missing filter.** A code path that queries the index without the tenant predicate. Usually a background job, an admin/debug endpoint, an eval harness, or a new feature written by someone who used the raw client. Mitigation: the tenant filter is injected by a wrapper, the raw client is not exported, and there is a test that asserts every query carrying no tenant predicate raises.

**2. The semantic cache.** This is the one that gets shipped. You cache `embedding(query) → results` or `query_text → answer` to save cost, and the cache key does not include the tenant. Tenant A asks "what is our Q3 revenue target," tenant B asks the same string, and B gets A's answer with A's citations. **The cache key must include the tenant ID and the full principal set**, which incidentally destroys most of the cache hit rate you were hoping for — which is itself an important finding about whether the cache is worth building.

**3. The reranker or embedding cache.** Same failure, one layer down, and easier to miss because it feels like a pure function. `rerank(query, doc) → score` is safe to cache keyed on the document ID *only if the document ID space is global and the score does not leak content*. Caching `embed(text) → vector` is generally safe, but the cache itself is now a store of tenant text and must be access-controlled like the index.

**4. Cross-tenant deduplication.** Someone adds content-hash dedup to save storage: identical chunks are stored once with a reference count. Now the *existence* of a document is shared, and depending on implementation, tenant B retrieving a chunk that only tenant A uploaded is a direct leak. Even done correctly, dedup leaks membership: "does tenant B have this exact document" becomes observable through timing or storage metrics.

**5. Shared fine-tuned models.** If you fine-tune an embedding model or a reranker on all tenants' data, tenant data is now in the weights. This is a memorisation risk that is hard to bound and impossible to un-ring; several enterprise contracts explicitly forbid it. If you must fine-tune, do it per-tenant or on synthetic/public data.

**6. Logs, traces and error payloads.** Retrieved chunk text in a trace span, a full prompt in an exception message, a request body in an access log. Your observability stack is now a copy of every tenant's documents with, typically, much weaker access controls than the primary store. This is the most common *real* leak I have seen, and it leaks to your own employees rather than to other tenants — which is still a contract violation.

**7. Eval and test datasets.** Someone builds a golden set from production queries and documents to fix a quality issue. That set now lives in a repo, gets shared with a vendor, or gets used to fine-tune. Treat eval data as production data.

**8. Global statistics and IDF.** In hybrid retrieval, BM25 uses corpus-wide document frequencies. If those are computed across tenants, a rare term's IDF reveals information about other tenants' corpora — a genuine but usually low-severity side channel. It also *hurts quality*, since a term that is rare globally may be common in this tenant's corpus.

**9. Document IDs and citation URLs.** Sequential IDs let a tenant enumerate your corpus size and, if any endpoint fetches by ID without an authorisation check, fetch other tenants' documents outright. Use opaque IDs and check authorisation on fetch, not just on search.

**10. The vectors themselves.** Covered in its own question below — embeddings are recoverable to a meaningful degree, so a vector store is a store of text, not a store of harmless numbers.

**⚠ Trap:** believing that "we filter by tenant on the vector query" is the security boundary. It is *one* of ten boundaries, and it is the one everybody implements. The leaks I have actually seen in production were the semantic cache and the trace payloads, both of which were added by well-meaning engineers doing performance and observability work, in PRs that no one thought to review as a security change. **The rule I enforce: any PR that adds a cache or a log line in the retrieval path is a security review, not a performance review.**

### Can someone recover text from an embedding? Does it change how I treat my vector store?

Yes, to a degree that should change your threat model, and this is a question I would expect a senior candidate to have a real answer to rather than a shrug.

The attack is embedding inversion. Given a target embedding and query access to the same embedding model, you can iteratively generate candidate texts, embed them, and optimise toward the target — a search in text space guided by the embedding distance. The strongest published version of this trains a model to do the inversion directly and refines iteratively.

**📄 Paper:** Morris, Kuleshov, Shmatikov & Rush (2023), *Text Embeddings Reveal (Almost) As Much As Text* — introduced `vec2text`, showing that a substantial fraction of short texts (32-token inputs) can be reconstructed **exactly** from their embeddings, and much of the semantic content recovered even when exact reconstruction fails. It replaced the prior assumption that embeddings were a privacy-preserving representation.

The practical qualifications matter and you should state them: reconstruction quality degrades with input length (a 500-token chunk is much harder than a 32-token sentence), and the attack requires access to the *same* embedding model — which for an API model means the attacker needs API access, and for an open-weights model means anyone. Also, even failed reconstruction typically recovers topic, entities and numbers, which is often the sensitive part anyway.

**What this changes, concretely:**

**Classify the vector store at the same sensitivity level as the document store.** Not lower. Same encryption at rest, same access controls, same audit logging, same retention policy, same deletion guarantees. The very common architecture where the documents are in a locked-down S3 bucket and the vectors are in a SaaS vector database with a shared API key is an inconsistency your security team will find.

**Deletion means deleting the vector.** If a document is deleted for a GDPR erasure request, the embedding is derived personal data and must go too. "We removed it from the document store but the vector is still in the index" is not compliant, and is also a retrieval bug (see the next chunk).

**Cross-border data residency applies to vectors.** If you cannot move a customer's documents to a US region, you cannot move their embeddings either.

**Third-party embedding APIs are data processors.** Sending text to an embedding API is a data transfer, subject to the same DPA and retention terms as any other. Check whether the provider retains inputs and for how long.

**🗣 Say this in the room:** "I treat embeddings as derived personal data, not as anonymised features — vec2text showed short texts can be reconstructed almost exactly from their embeddings given model access, and even partial inversion recovers entities and numbers. So the vector store gets the same classification, encryption, access control, residency and deletion guarantees as the document store, and an erasure request deletes the vector, not just the row."

### Someone left the company yesterday. Their document is still being quoted in answers. Walk me through the debug.

This is the most important incident in this section because it is the one that reaches a customer, and because the answer reveals whether someone has actually operated a retrieval pipeline.

**First, establish which of four layers is stale**, in this order, because each is cheap to check and they fail independently:

**Layer 1 — the source of truth.** Is the document actually deleted upstream? Half of these incidents turn out to be that the upstream system did a soft delete, or moved it to a trash folder that your connector still indexes, or the "delete" was a permission revocation and the doc still exists. Check the source first; you would be astonished how often the pipeline is innocent.

**Layer 2 — the ingestion pipeline.** Did a delete event get produced, and was it consumed? Deletes are the second-class citizen of every ingestion pipeline I have reviewed. Common root causes: the connector polls for *modified* documents and a deleted document simply stops appearing — there is no event at all, which means deletes are only detectable by a full reconciliation; or the delete event went to a DLQ and nobody watches that DLQ; or the pipeline processes creates and updates from one topic and deletes from another with different retry semantics.

**Layer 3 — the index.** Was the delete applied? Check by ID. If the vector is gone but results still contain it, you have a caching problem, not an index problem. If the vector is present, the delete never landed — go back to layer 2. If the vector is *tombstoned but still returned*, your engine's filter on deleted entries is broken or your reader is on a stale segment/snapshot.

**Layer 4 — everything downstream of the index.** The semantic cache holding a full answer with citations. A precomputed "related documents" table. A materialised summary index (RAPTOR-style tree nodes, per-document summaries) built from the deleted content — **this is the one people always miss.** If you built hierarchical summaries, deleting a leaf does not delete the summary that quoted it. A conversation-memory store that already extracted the fact. And the model's own context, if the session is long-lived.

**Now the design fix, because the interviewer wants prevention not just diagnosis:**

**Make deletion a first-class event with its own SLA and its own alert.** Not "the pipeline handles CRUD."

**Build a reconciliation job.** Poll-based connectors cannot see deletes, so you must periodically diff: enumerate document IDs in the source, enumerate IDs in the index, and delete the difference. Daily for most systems, hourly for sensitive ones. This job is the only thing that catches missed deletes, and it should emit `orphaned_documents_found` as a metric that alerts at any non-zero value. **This one job would have prevented most retrieval-of-deleted-content incidents I have seen.**

**Add a serving-time existence check.** Before chunks reach the model, batch-verify with the source of truth (or a fast tombstone service) that each document still exists and is still visible to this user. One RPC, ~10–15 ms, over-fetch by 20% so drops do not leave you short. This is the same late-binding check as the ACL design, and it converts "eventually consistent index" from a correctness problem into a latency problem.

**Propagate deletes to every derived structure.** Maintain an explicit lineage: document → chunks → vectors → summary nodes → cache entries. If you cannot enumerate everything derived from a document ID, you cannot delete it, and you should say so out loud during design rather than discovering it during a GDPR request.

**🔍 Failure taxonomy — retrieval of deleted content, ordered by frequency:** (1) poll-based connector cannot observe deletes at all — no event was ever produced; (2) delete event dead-lettered and unmonitored; (3) semantic/answer cache serving a pre-deletion result; (4) derived summary or hierarchical-index node still containing the content; (5) index accepted the delete but a read replica or snapshot is stale; (6) soft delete upstream that your connector does not interpret as a delete. Check in that order.

### How do you set and prove an ACL freshness SLA?

You cannot promise "instant," so you promise a number and you measure it. The structure I use has three parts: an asymmetric SLA, an end-to-end probe, and a hard backstop.

**The asymmetric SLA.** Revocations propagate within **60 seconds, p99**. Grants propagate within **15 minutes, p99**. Content updates within **5 minutes, p99**. Deletions of documents within **60 seconds, p99**, with a daily reconciliation backstop. Those specific numbers are negotiable; the asymmetry is not. Revocation and deletion are the paths where lateness is an incident; grants and updates are paths where lateness is a support ticket.

**How you actually hit the revocation number** is by not relying on the index at all. Index propagation involves a connector poll, a queue, a write, and a segment flush — you cannot get that to 60 seconds reliably at scale. So revocations go to a **separate fast path**: a deny-list (a Redis set, or a bloom filter plus an exact check) written synchronously by the permission-change webhook and consulted at query time by the retrieval client. The index catches up asynchronously and the deny-list entry expires once the index confirms. This is a well-understood pattern — it is the same shape as a JWT revocation list — and it is the only way to hit a tight revocation SLA on an eventually-consistent index.

**The end-to-end probe, which is how you *prove* it.** This is the part people never build and it is the whole answer to "prove." Run a synthetic canary continuously:

1. A probe service creates a document in each source system with a known unique marker string, granted to a test principal.
2. It polls a retrieval query for the marker, recording the time until first retrievable → **grant latency**.
3. It revokes access, then polls the same query as that principal, recording time until no longer retrievable → **revocation latency**.
4. It deletes the document and polls until not retrievable by anyone → **deletion latency**.
5. It emits all three as histograms, per source system, and alerts on p99 breach.

Run this every five minutes per connector. It costs almost nothing and it is the only measurement that actually covers the whole chain — connector, queue, index, cache, and serving filter — including the parts you forgot exist. **Every unit test you write about permissions tests your code; only this tests your system.**

**The hard backstop** is the late-binding check described earlier: verify the final ~20 chunks against the live permission service before they reach the model. With that in place, a blown freshness SLA degrades to "we retrieved something we then discarded," which costs latency and recall, not confidentiality. I would not ship enterprise retrieval without it, and I would say exactly that in a design review.

**⚠ Trap:** measuring propagation latency from "event received by our pipeline" rather than from "change made in the source system." The connector's poll interval is usually the dominant term — a 15-minute Drive poll makes your beautiful 30-second internal pipeline a 15-minute system — and measuring from event receipt hides exactly that. Measure from the action, in the source, using a probe that performs the action itself.
