### Explain HNSW from the ground up. What are the layers actually doing, and where does the skip-list analogy break?

HNSW is **greedy hill-climbing on a proximity graph, made robust by starting the climb from a coarse zoomed-out view and refining**. If you only remember one sentence, make it that. Everything else — layers, M, ef — is machinery to make greedy descent not get stuck.

Start with the flat version, NSW. Build a graph where each vector is a node connected to roughly its nearest neighbours. To search, pick an entry node, look at its neighbours, move to whichever is closest to the query, repeat until no neighbour improves. This works because embedding data is locally navigable. It fails in two ways: greedy gets trapped in local minima, and if all edges are short-range you take `O(N^(1/d))` hops to cross the space. NSW fixed the second by keeping some long-range links that arise naturally from inserting in random order.

HNSW fixes both by adding a **hierarchy**. Every node is assigned a maximum layer `l` drawn from a geometric distribution — `l = floor(-ln(uniform()) × mL)` with `mL = 1/ln(M)` — so layer 0 contains everything, layer 1 about `1/M` of nodes, layer 2 about `1/M²`, and the top layer usually has a single node. Search starts at that top node and does **greedy descent with beam width 1** through the sparse upper layers: the upper layers are a coarse map, and travelling one hop there covers enormous distance in the embedding space. When you can no longer improve at layer `l`, you drop to layer `l−1` using your current best node as the entry point. At layer 0 — and only at layer 0 — you switch to a **beam search** with beam width `efSearch`, maintaining a candidate heap and a result heap, which is what buys you robustness against local minima.

Where the skip-list analogy is right: the geometric layer assignment, the coarse-to-fine descent, the `O(log N)` expected hops. Where it is **misleading, and I have seen this cost people the answer**: a skip list is a total order and the descent is exact — you cannot take a wrong turn. HNSW is a graph over a metric space with no total order, greedy descent absolutely can take a wrong turn, and the *only* thing that recovers from it is the beam width at layer 0. That is why `efSearch` is the recall knob and why the layers alone do not give you accuracy — they give you *speed*. A common misreading is "more layers = more recall." No: more layers = faster convergence to a starting region. Recall comes from `efSearch` and from the quality of the layer-0 edges.

The second piece of real machinery is **neighbour selection with pruning**. When you insert a node and find its `efConstruction` nearest candidates, you do not simply keep the M closest. You apply a heuristic: accept a candidate `c` only if `c` is closer to the new node than to any already-accepted neighbour. This is a diversity rule, and it is what creates long-range links — it prevents all M edges from pointing into the same dense cluster, which would leave the node unreachable from other directions. Turn that heuristic off and you get a graph that looks fine by average edge length and has terrible recall on clustered data.

**📄 Paper:** Malkov & Yashunin (2016/2018), *Efficient and Robust Approximate Nearest Neighbor Search Using Hierarchical Navigable Small World Graphs* — added the multi-layer hierarchy and the diversity-based neighbour-selection heuristic on top of their earlier flat NSW graphs, and displaced LSH and tree methods as the default in-memory index.

### Write the HNSW layer-0 search routine for me.

The upper-layer descent is trivial (beam of 1); the part worth writing is the layer-0 beam search, because every parameter question follows from it.

```python
import heapq

def search_layer(graph, vectors, q, entry_points, ef, dist):
    """graph: {node: [neighbour ids]}; returns up to `ef` closest node ids."""
    visited = set(entry_points)
    # candidates: min-heap by distance (frontier to expand)
    # results:    max-heap by distance (best-so-far, negated distances)
    candidates = [(dist(q, vectors[e]), e) for e in entry_points]
    heapq.heapify(candidates)
    results = [(-d, e) for d, e in candidates]
    heapq.heapify(results)

    while candidates:
        d_c, c = heapq.heappop(candidates)
        worst = -results[0][0]
        if d_c > worst and len(results) >= ef:
            break                      # frontier is worse than everything we hold: stop
        for n in graph[c]:
            if n in visited:
                continue
            visited.add(n)
            d_n = dist(q, vectors[n])
            worst = -results[0][0]
            if d_n < worst or len(results) < ef:
                heapq.heappush(candidates, (d_n, n))
                heapq.heappush(results, (-d_n, n))
                if len(results) > ef:
                    heapq.heappop(results)   # evict current worst
    return [e for _, e in sorted((-d, e) for d, e in results)]
```

Three things to say out loud while writing it. The **termination condition** `d_c > worst and len(results) >= ef` is the whole efficiency story — you stop expanding as soon as the best unexplored candidate is worse than your current `ef`-th best, which is the standard best-first-search bound. The **`visited` set** is why HNSW's memory during query is `O(ef)` for heaps but `O(nodes touched)` for the visited set; production implementations use a versioned bitset over all N nodes rather than a Python set, because allocating a hash set per query at high QPS is a real allocator problem. And **`ef` bounds the result heap, not the candidate heap** — a subtle inversion that people get backwards on a whiteboard.

**🏋 Drill:** 20 minutes, unaided. Write `search_layer` above, then answer without looking: (1) what changes if you set `ef = 1`? (2) What is the asymptotic number of distance computations in terms of `ef` and `M`? (3) Why must `ef ≥ k`? Pass criterion: you say "ef=1 reduces it to pure greedy with no backtracking," "roughly `ef × M` distance computations since each expanded node costs M evaluations," and "because the result heap is capped at ef, so you cannot return more than ef items."

### What do M and efConstruction control, and how much memory does the graph actually cost?

`M` is the number of bidirectional edges kept per node per layer, and it is the **build-time** parameter that sets the ceiling on achievable recall. Layer 0 typically gets `2M` connections (called `M0` or `Mmax0` in most implementations) because layer 0 is where accuracy is decided; upper layers get `M`. Typical values are `M = 16` for general embeddings, `M = 32–48` for high-dimensional or hard datasets, `M = 8` when memory is tight and you accept a recall ceiling around 0.9.

**📐 Numbers you must know — graph overhead.** Derive it rather than memorise it. Each edge is a 4-byte node ID. Layer 0 holds `2M` edges per node, so layer 0 alone costs `2M × 4 = 8M` bytes per vector. The geometric layer distribution puts about `1/(M−1)` of the nodes above layer 0 with `M` edges each, which adds a small fraction more — under 10% for `M ≥ 12`. So the durable figure is **`~M × 2 × 4` bytes per vector**: `M = 16` → **128 bytes**, `M = 32` → 256 bytes, plus single-digit-percent for the upper layers and whatever per-node bookkeeping the implementation adds. Say it as "about 130 bytes per vector at M=16, and I'd measure the implementation's real overhead before sizing the box."

The key consequence: **at d=1536 the graph is 2% of the index and the vectors are 98%.** `6144 bytes vs 128 bytes`. So "HNSW uses a lot of memory" is almost always false as stated — HNSW uses a lot of memory *because it needs the full vectors resident to compute distances during traversal*. The graph is cheap. This matters because it tells you where to attack: quantizing the vectors (int8, binary) shrinks the index by 4–32×; tuning M does essentially nothing to the total.

`efConstruction` is the beam width used during *insertion* to find each new node's candidate neighbours. It has no effect at query time and no effect on memory. It buys graph quality: a higher `efConstruction` means each node's M edges are closer to its true M nearest neighbours, which raises the recall ceiling you can reach at any `efSearch`. Typical 100–200; 400+ for datasets where you have measured that it helps. Its cost is linear in build time.

**⚠ Trap:** tuning `M` and `efConstruction` by copying a blog post and then never re-measuring after your embedding model changed. Optimal `M` depends on the *intrinsic dimensionality* of your data, not on d. A 3072-dim embedding of a highly-templated corpus (support tickets, invoices) has low intrinsic dimension and is happy at `M=12`; a 768-dim embedding of diverse web text may need `M=32`. Swap the embedding model and your previously-tuned parameters are stale. The check is cheap: build at `M ∈ {12, 16, 32}` on a 1M sample and plot recall@10 vs latency at swept `efSearch`. Three builds, an afternoon, and you get a curve instead of a folk belief.

### efSearch is my only runtime knob. Walk me through tuning it and what the curve looks like.

`efSearch` is the beam width at layer 0: the size of the result heap during the final search. It must be ≥ `k`, and everything interesting happens between `k` and about `20k`.

The shape of the curve is the single most useful thing to know here, and it is **strongly concave**: recall rises fast and then saturates, while latency rises roughly linearly. Concretely, for a typical 1M-vector text index at `M=16`, you will see something like recall@10 of 0.82 at `ef=16`, 0.94 at `ef=48`, 0.975 at `ef=100`, 0.99 at `ef=250`, and 0.994 at `ef=500`. Latency over that same sweep goes roughly 0.25 ms → 0.6 ms → 1.2 ms → 2.8 ms → 5.5 ms. So the last **0.5 points of recall cost you 4.5× the latency**. Almost nobody should buy that.

The tuning procedure I actually run, and it takes an hour:

1. Build the flat ground truth for 1,000 sampled production queries.
2. Sweep `efSearch` over `{k, 2k, 4k, 8k, 16k, 32k}` and record `(recall@k, p50, p95)`.
3. Plot recall against p95, not against `ef`. The knee is visible and it is where you sit.
4. Then — and this is the step people skip — run the *end-to-end* eval (answer quality, or nDCG against labelled relevance) at three points on that curve. Very often the end-to-end metric saturates well before ANN recall does, because the reranker downstream is forgiving. If ANN recall 0.94 and 0.99 produce identical answer quality, you take 0.94 and pocket the 2 ms.

The operationally interesting move is **per-request `efSearch`**. It is a query parameter, not an index property. So: interactive typeahead gets `ef=32`, the main chat path gets `ef=100`, the "deep research" agent that will make forty retrieval calls gets `ef=200` on the first call and `ef=48` on follow-ups, and your nightly eval job gets `ef=1000` so its numbers are near-exact. In pgvector this is a session GUC (`SET hnsw.ef_search = 100`), which means you can set it per-transaction from your connection pool — exactly the same pattern as setting `statement_timeout` per workload class.

**🗣 Say this in the room:** "efSearch is a per-query knob, so I don't pick one value — I pick a recall/latency operating point per traffic class. I sweep it against a flat ground truth, plot recall against p95 rather than against ef, and then check whether the end-to-end quality metric even moves across that range. It usually saturates before ANN recall does, and that gap is free latency."

### Our HNSW build for 50M vectors is taking fourteen hours. Where is the time going and what would you do?

Build cost is `N` insertions, each of which is a **full HNSW search** with beam `efConstruction`, followed by the pruning heuristic and bidirectional edge repair on up to `M` neighbours. So it is `O(N × efConstruction × M × d)` distance computations — `O(N log N)` in the standard telling, but the constant is large and it is dominated by distance evaluations at full precision.

Ballpark the arithmetic. Each insertion at `efConstruction=200`, `M=16` touches roughly `ef × M = 3,200` nodes, each costing a d=1536 dot product = 1536 MACs. That is `4.9M` MACs per insertion. For 50M vectors: `2.5e14` MACs ≈ 0.5 PFLOP. That is not the problem — a GPU would eat it. The problem is that **HNSW insertion is a pointer-chasing, random-access, cache-hostile workload**: every neighbour lookup is a random read into a 307 GB vector array. You are bound by DRAM latency (~80 ns) and TLB misses, not by FLOPs. Fourteen hours for 50M is roughly 1,000 inserts/sec/core-group, which is depressingly normal.

What I would actually do, in order:

**1. Parallelise the build properly.** HNSW insertion parallelises well with per-node locks; most libraries expose a thread count and most people leave it at the default. Going from 4 to 32 threads is close to linear until you saturate memory bandwidth — call it 6× real. That alone takes 14 h to ~2.5 h.

**2. Lower `efConstruction` and measure whether you lost anything.** Dropping 200 → 100 halves the dominant term. Build a 2M-vector sample at both and compare the recall-vs-`efSearch` curves. In my experience the curves are indistinguishable above `efConstruction ≈ 128` for most text corpora; you are paying for nothing.

**3. Reduce the distance cost.** Build the graph on **int8-quantized or Matryoshka-truncated vectors**. If your embedding model supports Matryoshka truncation, building the graph on the first 512 dimensions cuts distance cost 3× and the graph you get is nearly as good — you then rescore with full vectors at query time. This is the single biggest lever and almost nobody pulls it.

**4. Stop rebuilding from scratch.** If the 14 hours is recurring, the real bug is your operational model. HNSW supports incremental insertion; a nightly full rebuild is a choice, usually made because deletes are accumulating (which we will get to) or because the embedding model changed (which is a genuine full-rebuild event, but should be rare).

**5. Shard.** Ten shards of 5M each build in parallel on ten machines in ~1/10 the wall clock, and query fan-out across 10 shards costs you one extra network hop plus a merge. The recall cost of sharding is *zero* if you query all shards and merge — you are just partitioning the search. Do not shard by a semantic key hoping to query one shard; shard randomly and fan out.

**💰 Math:** at 32 cores you are looking at ~2.5 h on a `c7i.8xlarge` at ~$1.43/hr = **$3.60 per full rebuild**. Ten shards on ten `c7i.4xlarge` at ~$0.71/hr for 30 minutes = `10 × 0.71 × 0.5 = $3.55` — same money, 5× the wall clock improvement. Build cost is essentially never the money; it is the wall clock, because wall clock is how long your reindex window is and therefore how long you carry two indexes.

### What actually happens when you delete a vector from an HNSW graph?

Nothing good, and this is the failure mode that nobody plans for, so it is the one I probe hardest.

You cannot remove a node from an HNSW graph cheaply. The node has up to `2M` incoming edges from layer 0 plus more from upper layers, and those neighbours chose it via the diversity heuristic — removing it does not just delete edges, it potentially disconnects regions that were reachable *only* through it. Properly repairing means, for every in-neighbour, re-running the neighbour-selection heuristic over its candidate set, which requires knowing that candidate set, which you did not store. So essentially every implementation does the same thing: **mark it deleted and keep it in the graph.**

The tombstone is a routing node. Traversal still goes through it — it still costs a distance computation and a random memory read — it just gets filtered out of the result set. Consequences, in the order they bite you:

**Recall silently degrades.** Your `efSearch` beam is a fixed size. If 30% of the nodes in the beam are tombstones, your *effective* `efSearch` is 0.7× what you configured, and recall drops accordingly. Nobody notices, because the query still returns k results — they are just worse ones. This is the silent failure. Your recall dashboard (if you built the flat shadow eval) catches it; your latency dashboard does not.

**Result-set truncation.** Worse: if you request k=10 and the beam surfaces 10 candidates of which 6 are tombstones, some implementations return 4. Others over-fetch and retry. Know which yours does, because "user asks a question and gets 4 chunks instead of 10" is a quality incident with no error log.

**Memory never comes back.** The vector, the graph edges, the ID mapping — all still resident. A corpus with 20% monthly churn has, after six months without compaction, roughly `1.2^6 ≈ 3×` the resident footprint of its live data. That is the index bloat line item on your bill.

**Latency creeps.** More nodes traversed per unit of useful recall.

**⚠ Trap:** the belief that "the vector database handles deletes." Every managed vector store accepts a delete API call and returns 200. What it does underneath ranges from "immediate tombstone, compaction on an opaque schedule you cannot observe" to "tombstone until you manually trigger an optimizer pass." The trap is that the *API is identical* across those behaviours, so your integration tests pass. The question to ask a vendor, verbatim: "after I delete 30% of my vectors, what is my recall and my resident memory, and what command makes it go back to normal?" If they cannot answer with a number, you own the compaction problem yourself.

**🔍 Failure taxonomy — the deletion-driven degradation ladder:**
1. Tombstone ratio > 10% → recall starts measurably sliding; alert here.
2. > 25% → result truncation begins on high-k queries; memory is 1.3× live data.
3. > 40% → p99 latency inflation becomes visible; some implementations start failing to find k results at all.
4. Mitigation order: raise `efSearch` (buys time, costs latency), run the engine's compaction/optimize if it has one, segment-level rebuild (rebuild only segments above a tombstone threshold — this is what Lucene-style segment merging does and why Elasticsearch handles this more gracefully than a monolithic HNSW), full rebuild with alias swap.

The design rule I enforce: **if your corpus has any deletion at all, you must have a scheduled compaction job and a tombstone-ratio metric before you go to production.** Not after. The systems that get this right are the ones built on immutable segments plus background merge — the Lucene model — because they inherited a solved compaction story from a different problem.

### What is ScaNN's anisotropic quantization, and why does it beat plain PQ for inner-product search?

The insight is beautiful and it is a genuine "why didn't I think of that." Plain PQ minimises **reconstruction error**: it wants `||x − x̂||²` small, treating every direction of error as equally bad. But you are not trying to reconstruct `x`. You are trying to preserve `q · x` well enough to *rank* correctly. Those are different objectives, and the difference has structure.

Decompose the quantization error into two components relative to the datapoint's own direction: the **parallel** component (along `x`) and the **orthogonal** component. For maximum inner-product search, the queries that matter for a given datapoint `x` are the ones roughly aligned with `x` — those are the queries for which `x` is a top candidate. For such a query, the error in `q · x̂` is dominated by the *parallel* error, because the orthogonal error contributes only through the small orthogonal component of `q`. Meanwhile for a query far from `x`'s direction, you do not care about the error at all — `x` was never going to be in the top-k.

So ScaNN's **anisotropic loss weights parallel error more heavily than orthogonal error**, with the weighting derived from the distribution of query-datapoint angles. Concretely, the training objective becomes a weighted sum `η · ||e_parallel||² + ||e_orthogonal||²` with `η > 1`, and you fit the codebooks to that instead of plain squared error. The practical effect is that vectors with large norms — precisely the ones that dominate inner-product rankings — get their magnitudes preserved much more faithfully than PQ would preserve them.

This directly fixes the mismatch I flagged earlier: PQ is correct for L2 (and for cosine on normalised vectors) but systematically mis-ranks under raw inner product, because squared-error quantization treats a large-norm vector and a small-norm one identically while MIPS does not. Anisotropic quantization is PQ with the loss corrected for the metric you actually use.

**📄 Paper:** Guo, Sun, Lindgren, Geng, Simcha, Chern & Kumar (2020), *Accelerating Large-Scale Inference with Anisotropic Vector Quantization* (ICML) — introduced the score-aware, direction-weighted quantization loss and shipped it as ScaNN, replacing reconstruction-error PQ as the strongest quantizer for MIPS.

The practical situation: ScaNN is genuinely excellent and consistently near the top of the ann-benchmarks leaderboard, and it is what backs Google's vector search products. Its adoption friction is that it is a Google library with a narrower ecosystem than FAISS, and its build is less forgiving. **My decision rule:** if you are on a Google stack or you have a MIPS workload with unnormalised vectors and you are memory-constrained, benchmark ScaNN. Otherwise the marginal gain over OPQ-IVF-PQ or a good HNSW is usually smaller than the integration cost. Where I would definitely mention it in an interview is when someone asks "is PQ the state of the art for compression" — the answer is no, and knowing *why* (metric-aware loss beats reconstruction loss) is the signal.

### Explain DiskANN and the Vamana graph. What is the per-query IO budget?

DiskANN answers a specific question: **how do you serve a billion vectors from one machine when the vectors are 6 TB and your RAM is 256 GB?** The answer is a memory/disk split that is much smarter than "just page the index in."

**Vamana** is the graph. It differs from HNSW in two ways worth naming. It is **single-layer** — no hierarchy — which matters because a hierarchy means multiple random reads per query at different levels. And its pruning rule is a **relaxed** version of HNSW's diversity heuristic, parameterised by `α ≥ 1`: a candidate `c` is pruned only if some already-selected neighbour `p` satisfies `α · dist(p, c) ≤ dist(x, c)`. With `α = 1` this is exactly HNSW's rule; with `α = 1.2` you keep more long-range edges, which makes the single-layer graph navigable without a hierarchy. Building typically runs two passes, `α=1` then `α=1.2`, and the result is a graph with a **shorter path length** than HNSW's layer-0 graph — which is the whole point, because on disk each hop is an IO.

The layout is the key engineering. **In RAM**: compressed (PQ) representations of all N vectors, plus the entry point. **On SSD**: for each node, a single contiguous block containing its full-precision vector *and* its adjacency list, page-aligned. So one 4 KB random read gives you both "how far is this node really" and "where can I go next" — no pointer chase to a separate edge array.

Search: do greedy beam search using the **in-memory PQ distances** to decide where to go, which is fast and requires no IO. For each node you actually visit, issue one SSD read to get its true vector and neighbours. Maintain a small set of the best candidates by *exact* distance from those reads, and return those. So the PQ codes drive navigation and the SSD reads provide accuracy — a two-tier scheme where the cheap approximate representation does the searching and the expensive exact one does the scoring.

**IO budget.** A typical Vamana search visits on the order of 50–150 nodes at 0.9+ recall@10. At one 4 KB read each, that is **200–600 KB and 50–150 IOPS per query**. An NVMe drive doing 500k IOPS therefore supports roughly `500,000 / 100 = 5,000 QPS` from the drive alone, at a per-query latency of `~100 reads × 80 µs / queue_depth`. With io_uring at depth 32 and beam-parallel reads, you land around **2–5 ms p50 and 10 ms p99** — an order of magnitude worse than in-memory HNSW's 0.5 ms, and 100× cheaper per vector stored.

**📄 Paper:** Subramanya, Devvrit, Kadekodi, Krishnaswamy & Simhadri (2019, NeurIPS), *DiskANN: Fast Accurate Billion-point Nearest Neighbor Search on a Single Node* — introduced the Vamana graph and the RAM-PQ/SSD-full-vector split, making billion-scale single-node ANN practical where previous work required a cluster.

**💰 Math — this is the argument you make to your VP.** 1B vectors at d=1536. In-memory HNSW needs `6.14 TB + 128 GB graph ≈ 6.3 TB` of RAM: seven 1 TB-RAM instances at ~$8/hr = `7 × 8 × 730 = $40,880/month`. DiskANN needs the PQ codes resident — at `m=192`, `1e9 × 192 = 192 GB` — plus 6.3 TB on NVMe. One `i4i.8xlarge` class node (256 GB RAM, 7.5 TB NVMe) at ~$2.75/hr = `2.75 × 730 = $2,008/month`. **That is a 20× cost reduction for roughly 3–5 points of recall and 5× the latency.** Whether you take that trade is a product decision, but you must be able to put both numbers on the whiteboard. **📅 Volatile:** instance types and prices; re-derive.

### How does SPANN differ from DiskANN, and when would you reach for it?

Both are SSD-resident billion-scale systems, but they make **opposite structural bets**, and being able to state the difference cleanly is a good senior signal.

DiskANN is a **graph** on disk: navigation state in RAM (PQ codes), one random read per hop. SPANN is an **inverted-file** on disk: the centroids live in RAM as a small in-memory ANN index, and each posting list is a contiguous block on SSD. Query: search the in-memory centroid index to pick which posting lists to open, then do a small number of *large sequential* reads to pull those lists, then scan them.

That difference is an IO-pattern difference and it is decisive. DiskANN issues ~100 small random 4 KB reads. SPANN issues a handful of large sequential reads. **Random 4 KB reads are IOPS-bound; large sequential reads are bandwidth-bound.** On NVMe both are fine, but on network-attached or object storage — where a request costs 20–100 ms of latency and bandwidth is plentiful — SPANN's pattern is dramatically better. This is why the object-storage-native vector stores look structurally like SPANN and not like DiskANN.

SPANN's two named contributions solve the problems that a naive disk-IVF would have. First, **balanced partitioning with multiple assignment**: it uses a hierarchical balanced clustering so posting lists are similar in size (remember the IVF imbalance failure mode), and it assigns boundary vectors to *several* posting lists rather than one, so a query near a Voronoi boundary still finds its neighbour without opening many lists. That closes IVF's structural recall leak at the cost of some duplication. Second, **query-dependent pruning**: it decides how many lists to open based on the distance ratio between the closest centroid and the others, rather than a fixed `nprobe` — so easy queries open one list and ambiguous ones open eight, which is the right adaptive behaviour.

**📄 Paper:** Chen et al. (2021, NeurIPS), *SPANN: Highly-efficient Billion-scale Approximate Nearest Neighbor Search* — a memory-disk hybrid inverted index with balanced multi-assignment partitioning and adaptive list selection, offering an alternative to graph-on-disk designs like DiskANN.

**My decision rule:** local NVMe and latency-sensitive → DiskANN family (and note that pgvectorscale's StreamingDiskANN is exactly this, inside Postgres). Object storage, elastic/serverless, cost-dominant, latency budget in the tens of milliseconds → SPANN-shaped. And notice this is the same decision you already make between a B-tree on local SSD and a columnar file on S3: **random-small versus sequential-large is the axis, and the storage medium picks the winner.**

### Explain RaBitQ and where binary quantization sits against PQ.

Binary quantization is the extreme end of compression: keep one bit per dimension. Naively, `b = sign(x)`, and distance becomes Hamming distance, computed as `popcount(a XOR b)` — at d=1536 that is 24 × 64-bit XOR-and-popcount instructions, a handful of nanoseconds, over 192 bytes instead of 6144. **32× compression and a distance computation that is essentially free.** The problem with naive SimHash-style binarization has always been that it loses badly on accuracy and, worse, gives you no way to reason about how much you lost.

**RaBitQ's contribution is a binarization with an unbiased distance estimator and a provable error bound.** The construction: apply a random orthogonal rotation to the data (which normalises the coordinate distribution without changing distances), then quantize each rotated coordinate to one bit, and — critically — store a small number of extra scalars per vector (the norm and a correction factor) so that you can form an *unbiased estimate* of the true inner product from the bit code rather than just a monotone proxy. Because the estimator is unbiased with a bounded variance that shrinks as `1/sqrt(d)`, high-dimensional vectors quantize *better*, which is a pleasant inversion of the usual curse-of-dimensionality story.

**📄 Paper:** Gao & Long (2024), *RaBitQ: Quantizing High-Dimensional Vectors with a Theoretical Error Bound for Approximate Nearest Neighbor Search* (SIGMOD) — gave 1-bit-per-dimension quantization a rigorous unbiased estimator with error bounds, making binary codes competitive with PQ at far lower distance-computation cost. Extended versions add multi-bit variants that trade compression for accuracy on a smooth curve.

Where it sits against PQ, honestly:

- **Compression**: RaBitQ at 1 bit/dim gives `4d/(d/8) = 32×`. PQ at `m = d/8` gives the same 32×. Comparable.
- **Distance speed**: RaBitQ wins decisively. Hamming/popcount over 192 bytes beats a 192-entry LUT gather, because the LUT gather is a dependent random access chain and popcount is pure SIMD with no data-dependent addressing.
- **Build cost**: RaBitQ wins outright — no k-means, nothing to train. The rotation is a fixed random orthogonal matrix; encoding is a matmul and a sign. Encoding 100M vectors is one `[1e8, 1536] @ [1536, 1536]` product = `1e8 × 1536 × 1536 × 2 = 4.7e14` FLOP, about **3 seconds** of A100 compute at 150 TFLOP/s effective — so it is entirely bound by reading the 614 GB of source vectors, not by arithmetic. PQ requires codebook training plus a full assignment pass.
- **Accuracy without rescoring**: PQ is generally better, because 8 bits per subspace of 8 dims carries more information than 8 bits for 8 dims one-apiece.
- **Accuracy with rescoring**: they converge, which is why the rescoring step is the actual production requirement in both cases.

**📅 Volatile:** this is the fastest-moving corner of the whole section. Qdrant, Milvus and several others have shipped binary-quantization modes, pgvector added a `bit` type with Hamming distance, and RaBitQ-family variants are landing in engines through 2025–2026. Do not assert what any specific engine implements today without checking; assert the mechanism and say "I'd verify which variant this engine ships."

### Give me the arithmetic for binary quantization plus float rescoring. How much recall do I get back?

This is the pattern I would default to for any index above ~50M vectors, so the arithmetic should be reflexive.

**Storage.** d=1536. Binary: `1536 bits = 192 bytes`. Full float32: 6144 bytes. **32× compression.** For 100M vectors: 19.2 GB of bit codes instead of 614 GB of floats. The bit codes fit in RAM on a commodity box; the floats live on NVMe or object storage.

**Search.** Build the HNSW graph over the *binary* codes with Hamming distance. Traversal distance computations are now 24 popcounts instead of 1536 float multiply-adds — call it 30–50× cheaper per hop, and the memory traffic during traversal drops 32×, which matters more since traversal is random-access. Retrieve an over-fetched candidate set of size `R`.

**Rescore.** Fetch full float vectors for the `R` candidates and compute exact scores. `R = 200`: `200 × 6144 = 1.2 MB` of random reads. From RAM (if you keep floats resident) it is free; from NVMe it is 200 IOPS at ~80 µs each, ~1 ms at queue depth 16; from S3 it is 200 GETs, which you must batch or you will eat 200 × 30 ms — this is why object-storage designs store vectors in contiguous, co-located blocks.

**Recall.** The honest published pattern, and I would state it with the hedge: binary-only recall@10 typically lands somewhere in the 0.6–0.8 range depending on the embedding model, and rescoring with an over-fetch of **`R = 10k` to `20k`** (so 100–200 candidates for k=10) recovers it to roughly 0.93–0.97. Two structural caveats. First, **the recovery depends entirely on the embedding model** — models trained with binary/Matryoshka-aware objectives (some Cohere and Nomic releases advertise this explicitly) binarize far better than models that were not. Second, **rescoring can only reorder what binary search nominated**; if the true nearest neighbour never entered the candidate set, no amount of rescoring finds it. So the over-fetch factor is the real recall knob, not the rescoring itself.

**💰 Math — the whole reason to do this.** 100M vectors. All-float in-memory HNSW: 614 GB vectors + 12.8 GB graph = 627 GB → a 768 GB-RAM instance at ~$6/hr = **$4,380/month**. Binary + NVMe rescore: 19.2 GB codes + 12.8 GB graph = 32 GB RAM, floats on local NVMe → a `c7gd.4xlarge`-class node at ~$0.73/hr = **$533/month**. That is **8.2× cheaper for a few points of recall**, and you can buy some of those points back by raising the over-fetch, which costs IOPS not RAM. This is the single highest-leverage cost decision in vector serving right now.

**⚠ Trap:** binarising an embedding model that was not trained for it and shipping without a recall measurement. I have seen a team take a 32× storage win on paper and ship a 0.71-recall index because their embedding model's information was concentrated in a few high-magnitude dimensions that one bit each could not represent. **The check is one afternoon**: binarise a 1M sample, compute recall@10 against flat with and without rescore at `R ∈ {50, 100, 200, 500}`, and look at the curve. If rescore-at-200 does not get you past 0.9, your model is not a binarization candidate — try int8 scalar quantization (4× compression, near-lossless for almost every model) instead.

### I have 30M vectors, 5% churn per day, a 50 ms p99 budget, and a filter on every query. HNSW, IVF-PQ or DiskANN — pick one and defend it.

I would take **HNSW with int8 scalar quantization in an engine with native filtered search**, and here is the reasoning chain, because the reasoning is what is being graded.

**Start with the constraint that eliminates options.** Memory: `30e6 × 1536 × 4 = 184 GB` raw. That is affordable but not comfortable. At int8 scalar quantization it is `30e6 × 1536 = 46 GB`, plus an HNSW graph at `M=16` of `30e6 × 128 = 3.8 GB` — call it **50 GB, one 64 GB box**. Int8 SQ is near-lossless for text embeddings (typically <1 point of recall) because you are quantizing per-dimension with a learned min/max, not throwing away structure. So the memory argument does not force me off HNSW at this scale. If it were 300M, it would.

**Now the churn.** 5% per day of 30M = **1.5M writes/day = 17 writes/sec sustained**. That is nothing for insert, but it is the tombstone problem: after 20 days without compaction I have accumulated 30M tombstones — a 100% bloat ratio. So my design *must* include segment-based storage with background merge, or a scheduled rebuild. This is exactly why I want an engine that does Lucene-style immutable segments plus merge (Elasticsearch/OpenSearch, Vespa, Qdrant's segment optimizer) rather than a single monolithic in-process HNSW. **The churn requirement is what picks the engine, not the algorithm.**

**Now the filter, which is the deciding factor.** A filter on every query kills naive HNSW, because you either post-filter (recall collapse, which we will do the arithmetic on) or pre-filter into a brute-force scan. I need an engine implementing **filtered graph traversal** — Qdrant's filterable HNSW with payload indexes, or Vespa, or a Filtered-DiskANN-style approach. This is the single most important selection axis for this workload and it eliminates several popular options.

**Why not IVF-PQ.** At 30M I do not need 30× compression, and IVF's centroid drift under 5% daily churn means retraining every few weeks. I would be paying an operational tax for a memory saving I do not need. IVF-PQ becomes right when raw storage exceeds affordable RAM — north of ~100M at this dimension.

**Why not DiskANN.** My p99 budget is 50 ms end-to-end, which sounds generous, but that 50 ms has to cover query embedding (~15–25 ms for an API call, or ~5 ms self-hosted), the ANN search, the rerank, and the network. Spending 5–10 ms of it on SSD round trips when 0.5 ms of RAM search is affordable is a bad trade. **DiskANN is a cost play, not a latency play**, and at 50 GB there is no cost to play for. Also: DiskANN-family indexes historically handle high churn worse than segment-merge designs, since the graph is optimised for a static layout.

**🗣 Say this in the room:** "At 30M and d=1536, int8 HNSW is 50 GB — one box, sub-millisecond. So memory doesn't force me to compress, which rules IVF-PQ and DiskANN out as unnecessary complexity. The two requirements that actually pick the design are the 5% daily churn, which demands segment-based storage with background merge so tombstones don't rot my recall, and the mandatory filter, which demands native filtered graph traversal rather than pre- or post-filtering. So: HNSW with int8 SQ, in an engine with payload indexes and a segment optimizer, and I'd hold flat-search ground truth on 1,000 sampled queries to watch recall daily."
