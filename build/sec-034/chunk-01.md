### Before we talk about any index — when is brute-force flat search the right answer, and what does it actually cost?

Flat search is not the naive thing you graduate from. It is **the recall ceiling and the only unbiased measurement instrument you have**, and for a surprising fraction of production systems it is also the correct final answer. Every ANN index is defined by how much recall it gives up relative to flat; if you never build flat, you literally cannot state your recall, because recall@k is defined against exact nearest neighbours. So flat is both the baseline and the ruler.

Mechanically it is one GEMM. You hold an `[N, d]` float32 matrix, L2-normalize both sides, and compute `scores = X @ q` for a single query or `X @ Q.T` for a batch, then `argpartition` for top-k. There is no graph, no training, no build step, no tombstone, no compaction, no drift. Insert is `append`. Delete is a mask.

Here is the number that decides the argument, and it is the same number that governs LLM decode, which is why you already have the intuition. **Flat search at batch size 1 is memory-bandwidth-bound, not compute-bound.** Take 1M vectors at d=1536 float32: `1e6 × 1536 × 4 = 6.14 GB`. One query is `1e6 × 1536 = 1.54e9` multiply-adds ≈ 3.07 GFLOP. A 16-core server doing ~1 TFLOP/s fp32 would finish the arithmetic in 3 ms — but it has to stream 6.14 GB out of DRAM at ~100 GB/s, which is **61 ms**. Arithmetic intensity is 0.5 FLOP/byte. This is a GEMV, and GEMVs are bandwidth-bound. Exactly like decoding one token.

And the fix is exactly the same fix: **batch**. Run 100 queries together and you read those 6.14 GB once and do 307 GFLOP against them — now it is a GEMM, compute-bound, ~0.3 s for the batch, i.e. **3 ms per query amortised**, a 20× improvement from batching alone. On an A100 with 1.5 TB/s HBM, the single-query read is `6.14 GB / 1.5 TB/s = 4.1 ms`.

**📐 Numbers you must know:** a d=1536 float32 vector is `1536 × 4 = 6144 bytes ≈ 6 KiB`. So **1M vectors ≈ 6 GB, 10M ≈ 61 GB, 100M ≈ 614 GB, 1B ≈ 6.1 TB** — all before any index overhead. Memorise the 6 KiB and derive the rest live; that one constant answers half of the capacity questions in this domain.

Where flat wins outright: under ~100k vectors (614 MB, ~6 ms scan single-threaded, sub-millisecond batched), where the corpus churns constantly and index build cost would dominate, where you need exact results for a compliance or eval reason, and — critically — **inside a per-tenant partition**. A B2B product with 50,000 tenants averaging 2,000 chunks each does not need a global ANN index; it needs 50,000 tiny flat scans, and each one is 12 MB and 0.1 ms. I have killed more than one Pinecone bill by noticing that the "billion vector" problem was actually fifty thousand independent small problems.

**🗣 Say this in the room:** "I'd start flat, because flat is the ruler — recall@k is undefined without it. At d=1536 that's 6 KiB per vector, so a million vectors is 6 GB and a single-query scan is a 6 GB DRAM read, roughly 60 ms bandwidth-bound. If my corpus is under a few hundred thousand vectors, or naturally partitions per tenant, I ship flat and spend the saved complexity budget on retrieval quality instead."

### Draw me the trade-off space on this whiteboard. What are the axes and where does each index family sit?

This is the question I would ask if I only got to ask one, because the answer reveals whether someone has tuned an index or read about indexes. There are **four** axes and every design conversation is a movement along them: **recall, query latency (or QPS), memory footprint, and build/update cost.** You get to pick three. That is the whole field.

Draw a 2×2 with recall on the vertical and memory on the horizontal, then annotate each point with latency and build time:

- **Flat**: recall 1.00 by definition, memory 1.0× the raw vectors, build time zero, updates free, latency terrible and linear in N. Top-left of the quality axis, worst on latency.
- **IVF-Flat**: recall 0.90–0.99 tunable at query time via `nprobe`, memory ~1.0× (you still store full vectors, plus a tiny centroid table), build = one k-means pass, latency ~`nprobe/nlist` of flat. Cheap to build, easy to reason about, degrades as the data distribution drifts away from the trained centroids.
- **IVF-PQ**: recall 0.70–0.95 depending on code length and whether you rescore, memory 0.03–0.25× (this is the axis it wins on, by a lot), build = k-means plus codebook training, latency very good because distances are table lookups over bytes. This is what you use when the vectors do not fit in RAM.
- **HNSW**: recall 0.95–0.99+ at excellent latency (sub-millisecond at 1M scale), memory ~1.0× the raw vectors **plus** the graph, build expensive (`O(N log N)` insertions each doing a full search), and deletes are the ugly part. Best recall-per-millisecond in RAM; worst memory story of any graph method.
- **DiskANN / Vamana**: recall 0.90–0.97, memory ~0.05× in RAM (compressed vectors only) with the graph and full vectors on NVMe, latency ~1–10 ms dominated by SSD round trips, build very expensive. This is the billion-scale-on-one-box point.
- **Binary / RaBitQ + rescore**: recall 0.85–0.95 after rescoring, memory 0.03×, latency excellent (Hamming distance is popcount), build cheap. The newest and fastest-moving corner.

The trap in how people draw this is treating recall as a property of the index. **Recall is a property of the index *and the query-time parameter*, and you move along the curve at request time.** `nprobe` in IVF and `efSearch` in HNSW are runtime knobs; you can serve the same index at recall 0.85/2 ms for the autocomplete path and recall 0.99/40 ms for the "deep research" path. Candidates who present a single recall number for an index have never operated one.

**⚠ Trap:** optimising recall@k against exact-nearest-neighbour ground truth when your product cares about answer quality. ANN recall is an *intermediate* metric. I have seen a team spend three weeks moving HNSW recall@10 from 0.94 to 0.985 and measure zero change in end-to-end answer correctness, because the reranker downstream was re-sorting the top 100 anyway and the "missed" neighbours were near-duplicates of ones that were found. Always chain the measurement: ANN recall → retrieval quality (nDCG/recall@k against *labelled* relevance) → answer quality. Only the last one pays.

**🏋 Drill:** 10 minutes, no notes. Draw the four axes. Place flat, IVF-Flat, IVF-PQ, HNSW, DiskANN. For each, write the one runtime knob and the direction it moves recall and latency. Pass criterion: you name `nprobe`, `efSearch`, and "number of rescore candidates" without looking, and you can state which two families need a training step before you can insert the first vector.

### Define recall@k for an ANN index precisely, and tell me how you measure it in production where you have no ground truth.

Precision on the definition matters here because two different quantities get called "recall" and interviewers use the confusion as a filter.

**ANN recall@k** is `|A ∩ G| / k`, where `G` is the true top-k under the exact distance metric (computed by flat search) and `A` is what your index returned. It measures only the index's fidelity to brute force. It says nothing about whether the true top-k were *relevant*. **Retrieval recall@k** is `|retrieved ∩ relevant| / |relevant|` against human-labelled relevance judgements, and that is the one that correlates with your product. When someone says "our recall is 0.97" always ask which.

Measuring ANN recall without ground truth is a sampling problem, and the solution is a **shadow flat index over a sample**. You do not need exact neighbours for all 100M vectors; you need them for the queries you actually serve. So: log 1–2% of production queries with their returned IDs, and asynchronously — off the request path, on a batch worker at night — compute exact top-k for those queries by brute-forcing the full corpus. At 1,000 sampled queries against 100M × 1536 vectors, that is a `[1000, 1536] @ [1536, 100e6]` problem = `1000 × 1536 × 1e8 × 2 = 3.07e14` FLOP ≈ 307 TFLOP. On one A100 at ~150 TFLOP/s bf16 effective, that is about **35 minutes of GPU time, roughly $1.50 at $2.50/GPU-hour** — a rounding error, and you get a daily recall number with a real confidence interval.

The confidence interval matters. 1,000 queries × k=10 gives 10,000 Bernoulli trials, so the standard error on a recall of 0.95 is `sqrt(0.95 × 0.05 / 10000) ≈ 0.0022`. You can detect a 1-point recall regression with confidence. If you only sample 50 queries you cannot, and your "recall dashboard" is noise you will chase.

The cheap continuous proxy, for when you cannot afford even that: track the **distance distribution of returned results**. Log the mean and p90 of the top-1 and top-10 similarity scores per query bucket. Recall degradation from graph corruption, tombstone accumulation, or centroid drift shows up as the returned distances getting systematically worse — you are still finding *something*, just something further away. This is a leading indicator and it costs nothing.

**🔍 Failure taxonomy — "recall dropped and I don't know why," as a decision procedure:**
1. Did the *distance distribution* shift, or only the IDs? If IDs churned but distances are identical, you have ties/duplicates, not a regression.
2. Did corpus size grow >2× since the index was trained (IVF) or built (HNSW)? → centroid drift / graph degradation.
3. What fraction of the index is tombstoned? → deleted-node traversal cost and connectivity loss.
4. Did the embedding model version change for *writes* but not *reads*, or vice versa? → mixed-geometry index, and recall against the new queries is arbitrary.
5. Is a filter now more selective than it was (a new tenant, a new ACL rollout)? → post-filter collapse, which we will get to.
6. Only then suspect the index parameters, because they did not change on their own.

### Walk me through IVF from first principles. What is the coarse quantizer, and what do nlist and nprobe actually do?

IVF is **an inverted index where the "terms" are learned cluster IDs instead of words**. That framing makes it instantly familiar: in Lucene you post a document under every term it contains; in IVF you post a vector under the single centroid it is closest to. Query time, you decide which posting lists to open. The whole algorithm is "don't scan the vectors that live in neighbourhoods far from the query."

Build has two steps. First, run k-means over a training sample of your vectors to get `nlist` centroids — this is the **coarse quantizer**, and it is the only trained component. Then assign every vector in the corpus to its nearest centroid and append its ID (and, in IVF-Flat, its full vector) to that centroid's posting list. You now have `nlist` buckets averaging `N/nlist` vectors each.

Query has two steps that mirror it. Compute distances from the query to all `nlist` centroids — that is a tiny flat search, `nlist × d` work. Take the `nprobe` closest centroids, concatenate their posting lists, and do an exact scan over just those. Merge, top-k, done.

So: **`nlist` is a build-time knob that sets the granularity of the partition, and `nprobe` is the runtime recall/latency knob.** Scanned fraction ≈ `nprobe / nlist`, and speedup over flat ≈ `nlist / nprobe`, minus the centroid-scan overhead.

Sizing: the durable rule of thumb is `nlist ≈ sqrt(N)` for balance between the centroid scan and the posting-list scan, and you want enough training data that k-means is not garbage — roughly 40–100 vectors per centroid, so ≥ `40 × nlist` training vectors. For N = 10M: `sqrt(10e6) = 3162`, round to `nlist = 4096`. Each list holds `10e6 / 4096 ≈ 2441` vectors. At `nprobe = 32` you scan `32 × 2441 = 78,100` vectors, which is **0.78% of the corpus — a 128× speedup** — plus the 4096-centroid scan, which at d=1536 is `4096 × 1536 × 4 = 25 MB` read, ~0.25 ms. Note that the centroid scan is not free and becomes the floor on your latency; if you push `nlist` to 65536 chasing selectivity, the coarse scan alone is 400 MB and 4 ms.

Why does recall leak at all? Because the true nearest neighbour of a query can sit in a cell whose centroid is *not* among the `nprobe` closest — this happens whenever the query lands near a Voronoi boundary. The probability is highest for exactly the queries that are ambiguous, which is a nasty correlation: **IVF's misses are concentrated on the hard queries**, not uniformly distributed. That is why a global recall of 0.95 can still feel bad on the queries users complain about.

**⚠ Trap:** believing you can raise recall arbitrarily by raising `nprobe`. You can, and at `nprobe = nlist` you have reinvented flat search plus a wasted centroid scan. The useful operating range is `nprobe ∈ [8, 64]` for `nlist ≈ 4096`; past ~`nlist/16` you are paying flat-search prices for sub-flat recall. If you need `nprobe = 512` to hit your recall target, your problem is that `nlist` is wrong or your embedding space is not clusterable — go look at the cluster size histogram before you turn the knob further.

### Implement IVF-Flat from scratch — build and search — in numpy.

Roughly 40 lines, and I expect a candidate for a retrieval-heavy role to be able to write this cold.

```python
import numpy as np

class IVFFlat:
    def __init__(self, nlist=256, iters=10, seed=0):
        self.nlist, self.iters, self.rng = nlist, iters, np.random.default_rng(seed)

    def _kmeans(self, X):
        C = X[self.rng.choice(len(X), self.nlist, replace=False)].copy()
        for _ in range(self.iters):
            assign = np.argmax(X @ C.T, axis=1)          # cosine on unit vectors
            for j in range(self.nlist):
                members = X[assign == j]
                if len(members):
                    c = members.mean(0)
                    C[j] = c / (np.linalg.norm(c) + 1e-12)
        return C

    def build(self, X, ids):
        X = X / np.linalg.norm(X, axis=1, keepdims=True)
        self.C = self._kmeans(X)
        assign = np.argmax(X @ self.C.T, axis=1)
        # posting lists: list_id -> (vectors, original ids)
        self.lists = [(X[assign == j], ids[assign == j]) for j in range(self.nlist)]
        return self

    def search(self, q, k=10, nprobe=8):
        q = q / (np.linalg.norm(q) + 1e-12)
        probe = np.argpartition(-(self.C @ q), nprobe)[:nprobe]   # coarse step
        vecs = np.concatenate([self.lists[j][0] for j in probe])
        oids = np.concatenate([self.lists[j][1] for j in probe])
        if len(vecs) == 0:
            return np.array([]), np.array([])
        s = vecs @ q                                              # fine step
        top = np.argpartition(-s, min(k, len(s) - 1))[:k]
        top = top[np.argsort(-s[top])]
        return oids[top], s[top]
```

Two details interviewers poke at. First, `argpartition` not `argsort` — you want `O(N)` selection, not `O(N log N)` sort, and then you sort only the k survivors. Second, the re-normalisation of centroids inside k-means: on the unit hypersphere the mean of unit vectors is not a unit vector, and if you forget to re-project, your "cosine" k-means quietly becomes a hybrid that biases toward dense regions. Both of those are the kind of thing that makes a whiteboard implementation read as real.

**🏋 Drill:** 25 minutes, no autocomplete. Write the above from memory, then extend `search` to accept a boolean mask over `ids` and apply it *inside* the posting-list scan rather than after top-k. Pass criterion: your masked version returns exactly k results for a filter that keeps 1% of the corpus, and you can state in one sentence why the post-hoc version does not.

### My IVF index was trained on the first million vectors. We're at twenty million now and recall is sliding. What happened?

**Centroid drift**, and it is the structural weakness of IVF that HNSW does not share. The coarse quantizer is a *frozen model* fit to a *sample* of a distribution. Your corpus distribution moves — new customers with new document types, a new language, a product launch that adds a whole cluster of vocabulary — and the centroids stay where they were. Two things degrade simultaneously.

First, **imbalance**. Vectors from the new distribution do not spread evenly over the old Voronoi cells; they pile into whichever few cells happen to be nearest. I have seen a `nlist=4096` index where the largest posting list held 4% of the corpus. When `nprobe` opens that list you scan 800k vectors instead of the expected 5k and your p99 goes through the roof — this is a *latency* failure that looks like a recall failure because the timeout truncates the scan.

Second, **boundary error grows**. Recall depends on the true neighbour living in one of the `nprobe` nearest cells. When the cells no longer reflect the data, the boundary-crossing probability rises, and again it rises fastest for the newest content — so your recall regression is invisible on your old eval set and painfully visible to users asking about last month's docs.

The diagnostic is one query and it should be on a dashboard permanently: **the posting-list size histogram.** Emit `max/median` list size and the fraction of vectors in the top-1% of lists. Healthy is `max/median < 5`. When it crosses 10, retrain.

The fix ladder, cheapest first: (1) raise `nprobe` — buys you months, costs latency linearly; (2) retrain the quantizer on a fresh stratified sample and **reassign** — this requires reading and re-bucketing all N vectors but *not* re-embedding, so it is IO-bound not GPU-bound; (3) move to a graph index that has no global trained structure. The honest framing for an interview is: **IVF's build is cheap but its build is a model, and models go stale. HNSW's build is expensive but there is nothing to go stale — the graph is built from the data it contains.** That single sentence is the whole IVF-vs-HNSW argument for a mutating corpus.

**💰 Math:** reassignment cost for 20M × 1536 fp32. You read 20e6 × 6144 = 123 GB, compute 20e6 × 4096 centroids × 1536 dims = 1.26e14 MACs = 252 TFLOP. On one A100 at ~150 TFLOP/s effective that is 28 minutes of compute; the 123 GB read from object storage at 1 GB/s is 2 minutes. So a full IVF retrain+reassign for 20M vectors is **under an hour and about $1.50 of GPU** — versus re-embedding those same vectors, which we will price later at four orders of magnitude more. Retraining the quantizer is cheap. Never confuse it with reindexing.

### Explain product quantization. How do you get the compression, and what exactly are you throwing away?

PQ is **lossy compression of a vector into a short byte string, designed so that you can compute approximate distances directly on the compressed form without decompressing.** That last clause is the entire point and the reason PQ beats a generic compressor: you never materialise the original vector during search.

Mechanism. Split the d-dimensional vector into `m` contiguous subvectors of length `d/m`. For each of the `m` subspaces independently, run k-means with `2^nbits` centroids — almost always `nbits=8`, so 256 centroids per subspace. Now any vector is encoded as `m` bytes: the index of the nearest centroid in each subspace. A d=1536 vector with `m=192` becomes 192 bytes instead of 6144 — **32× compression**. The codebook itself is `m × 256 × (d/m) × 4 bytes = 192 × 256 × 8 × 4 = 1.57 MB`, negligible and shared across the whole corpus.

The search trick is **ADC — asymmetric distance computation.** The query stays in full precision. At query time you precompute a lookup table `LUT[j][c] = distance(q_subvector_j, centroid_c)` for all `m × 256` combinations — that is `192 × 256 = 49,152` tiny distance computations, done once per query. Then the approximate distance to any encoded vector is just `sum over j of LUT[j][code_j]`: **192 table lookups and 191 adds, no multiplies, over 192 bytes of memory**. That is why PQ is fast: you have converted a 1536-dim float dot product into a 192-byte gather-and-sum, an 8× reduction in bytes touched and a total elimination of floating-point multiplies from the inner loop. Bandwidth-bound workload, 32× less bandwidth.

What you throw away is **reconstruction error**, and its structure matters. Each subspace's error is the k-means quantization error in that subspace; total squared error is the sum across subspaces. The consequences: (a) distances get noisy, so the *ranking* near the top gets scrambled even when the right candidates are retrieved; (b) the error is roughly uniform across the space, so it hurts most when true neighbours are close together — i.e. on dense corpora with many near-duplicates, exactly the corpus you have.

On the compression ratio: it is exactly `4d / m` bytes-to-bytes at 8 bits. The commonly quoted "4–8× with negligible loss" corresponds to conservative settings with only 1–2 dimensions per subquantizer; in practice for text embeddings I see `d/m` of 4–8 dimensions per subvector, giving 16–32×, and then you **rescore**. Rescoring is not optional at those ratios and it is the thing juniors leave out: retrieve top-`R` (say 200) by PQ distance, then load the full float vectors for just those 200 and re-rank exactly. 200 × 6144 bytes = 1.2 MB of random reads — trivial — and it restores most of the lost ranking precision because PQ's job was only to *nominate* candidates, not to order them.

**⚠ Trap:** reporting recall on a PQ index without saying whether you rescored. PQ-only recall@10 at 32× compression might be 0.72; the same index with rescore-from-200 might be 0.94. Those are the same index. If a vendor benchmark does not state the rescore depth, the number is not comparable to anything.

**📄 Paper:** Jégou, Douze & Schmid (2011), *Product Quantization for Nearest Neighbor Search* — introduced the subspace-codebook decomposition and asymmetric distance computation, replacing binary-hash methods (LSH/spectral hashing) that had far worse distance fidelity per bit.

### Write me the PQ encoder and the ADC search loop.

```python
import numpy as np

class PQ:
    def __init__(self, m=8, nbits=8, iters=15, seed=0):
        self.m, self.K, self.iters = m, 2 ** nbits, iters
        self.rng = np.random.default_rng(seed)

    def _kmeans(self, X):                     # plain Lloyd's, one subspace
        C = X[self.rng.choice(len(X), self.K, replace=False)].copy()
        for _ in range(self.iters):
            a = np.argmin(((X[:, None, :] - C[None]) ** 2).sum(-1), axis=1)
            for j in range(self.K):
                if (a == j).any():
                    C[j] = X[a == j].mean(0)
        return C

    def fit(self, X):
        d = X.shape[1]
        assert d % self.m == 0
        self.ds = d // self.m
        self.codebooks = np.stack([                       # [m, K, ds]
            self._kmeans(X[:, i * self.ds:(i + 1) * self.ds]) for i in range(self.m)
        ])
        return self

    def encode(self, X):                                   # -> uint8 [N, m]
        out = np.empty((len(X), self.m), dtype=np.uint8)
        for i in range(self.m):
            sub = X[:, i * self.ds:(i + 1) * self.ds]
            d2 = ((sub[:, None, :] - self.codebooks[i][None]) ** 2).sum(-1)
            out[:, i] = np.argmin(d2, axis=1)
        return out

    def search(self, codes, q, k=10):                      # ADC
        lut = np.stack([                                   # [m, K]
            ((self.codebooks[i] - q[i * self.ds:(i + 1) * self.ds]) ** 2).sum(-1)
            for i in range(self.m)
        ])
        dist = lut[np.arange(self.m), codes].sum(axis=1)    # [N] gather + sum
        top = np.argpartition(dist, k)[:k]
        return top[np.argsort(dist[top])]
```

(The `_kmeans` here is deliberately the naive `O(N·K·ds)` version so it fits on a whiteboard; in production you call FAISS, which does this with SIMD and a proper mini-batch k-means.)

The line worth pointing at is `lut[np.arange(self.m), codes].sum(axis=1)`. That is the entire ADC search: a fancy-indexed gather into an `[m, 256]` table using the uint8 codes as indices, then a row sum. No multiplication, no decompression. In a real implementation the LUT is 256 × m float32 = 8 KB for m=8 and **fits in L1 cache**, which is why PQ scan rates hit hundreds of millions of vectors per second per core.

**⚠ Trap:** using PQ with a metric it was not fit for. The codebooks minimise *squared Euclidean* reconstruction error in each subspace. That is correct for L2 and correct for cosine **only if you L2-normalise before encoding** (on the unit sphere, `||a-b||² = 2 - 2·a·b`, so L2 and cosine rank identically). It is *not* correct for raw inner product on unnormalised vectors, where magnitude carries signal and squared-error quantization systematically mis-ranks the large-norm vectors that MIPS cares most about. That specific mismatch is what ScaNN was built to fix.

### What is OPQ, and does it actually earn its extra build step?

PQ makes an assumption it never states: that splitting the vector into contiguous chunks of coordinates produces subspaces with comparable variance and no cross-subspace correlation. For learned embeddings that assumption is straightforwardly false. Embedding dimensions are neither independent nor equally scaled — a handful of coordinates carry disproportionate variance (this is the same anisotropy that makes raw embedding spaces occupy a narrow cone). If subspace 3 has ten times the variance of subspace 47, both get 256 centroids and subspace 3 eats almost all the reconstruction error.

**OPQ — optimized product quantization — learns an orthogonal rotation matrix `R` applied before splitting**, chosen to balance variance across subspaces and decorrelate them. Because `R` is orthogonal, `||Rx - Ry|| = ||x - y||`, so rotating changes nothing about the true distances; it only changes how well the product decomposition can approximate them. Training alternates: fix `R`, retrain the codebooks; fix codebooks, solve for the `R` that minimises reconstruction error (a Procrustes problem, solved by SVD). At query time you rotate the query once — one `d × d` matvec, `1536² = 2.36M` MACs, microseconds — and everything downstream is unchanged.

Does it earn it? For text embeddings, yes, and this is one of the few places where I will give a near-unconditional recommendation: **if you are already paying for PQ, pay for OPQ.** The typical gain is a few points of recall at fixed code length, or equivalently ~20–30% shorter codes at fixed recall, which is real money at billion scale. The cost is a longer training phase (the alternating optimisation over a few hundred thousand training vectors) and one extra `d × d` float matrix, `1536 × 1536 × 4 = 9.4 MB`, stored once. In FAISS this is the difference between an `IVF4096,PQ192` factory string and `OPQ192,IVF4096,PQ192`.

**📄 Paper:** Ge, He, Ke & Sun (2013), *Optimized Product Quantization* — added the learned rotation before subspace decomposition, and showed the original PQ's fixed coordinate split was leaving substantial accuracy on the table for non-isotropic data.

**📅 Volatile:** the practical calculus here is shifting. Newer 1-bit schemes with theoretical error bounds (RaBitQ and its descendants) are displacing OPQ-PQ for some workloads, and Matryoshka-trained embeddings let you truncate dimensions *before* quantising, which changes the arithmetic. Verify what your vector store actually implements before your loop rather than assuming the 2013 stack.

### Size an IVF-PQ index for 100M vectors at d=1536 on a single box. Show me the numbers.

Start with what does not fit: raw storage is `100e6 × 1536 × 4 = 614 GB`. HNSW would need all of that resident plus a graph, so HNSW on one box is off the table unless you buy a 1 TB-RAM instance. That single line of arithmetic is the reason IVF-PQ exists.

**Choose the codes.** I want ≥16× compression and I intend to rescore, so `d/m = 8` dimensions per subquantizer → `m = 192`, `nbits = 8`. Per vector: 192 bytes of codes + 8 bytes of external ID = 200 bytes.

- PQ codes: `100e6 × 192 = 19.2 GB`
- IDs: `100e6 × 8 = 0.8 GB`
- Codebooks: `192 × 256 × 8 × 4 = 1.57 MB` — noise
- OPQ rotation: `1536² × 4 = 9.4 MB` — noise
- Coarse centroids at `nlist = sqrt(100e6) ≈ 10,000`, round to 16384: `16384 × 1536 × 4 = 100 MB`

**Total resident ≈ 20.1 GB.** That fits with enormous headroom on a 64 GB box, or on a single 96 GB A100 if you want GPU search. From 614 GB to 20 GB is the 30× that makes the deployment possible.

**Latency budget.** `nlist = 16384`, so lists average `100e6 / 16384 = 6,104` vectors. At `nprobe = 48` you scan `48 × 6104 = 293,000` codes = `293,000 × 192 bytes = 56 MB`. At 100 GB/s that is **0.56 ms** of streaming, plus the coarse scan of 100 MB — wait, that coarse scan is now the dominant term at 1 ms. So I would either keep centroids in fp16 (50 MB, 0.5 ms) or use a two-level coarse quantizer (HNSW over the centroids), which is what FAISS's `IVF16384_HNSW32` does and what every serious billion-scale deployment uses. With that, coarse cost drops to microseconds and query latency is ~0.6 ms single-threaded before rescoring.

**Rescoring.** Take top-200 by ADC, fetch 200 full vectors from NVMe: `200 × 6144 = 1.2 MB` in 200 random 6 KB reads. An NVMe drive at 500k IOPS and ~80 µs latency, issued with io_uring depth 64, finishes in well under 1 ms. So end-to-end **~1.5–2 ms p50** for a 100M-vector index on one box, at maybe 0.93–0.96 recall@10.

**💰 Math:** compare deployments. IVF-PQ: one `r6i.4xlarge`-class box (16 vCPU, 128 GB) plus NVMe, call it ~$1.20/hr = **$876/month**. HNSW fp32: needs ~630 GB resident, so a 1 TB-RAM instance at roughly $8/hr = **$5,840/month**, 6.7× more, for maybe 3 points more recall. If those 3 points do not show up in your end-to-end answer-quality eval — and you must run that eval before you spend the money — the IVF-PQ box is the correct answer and you have found $60k/year. **📅 Volatile:** instance prices; re-derive from the current on-demand sheet.

### LSH was the textbook answer for a decade. Why did it lose, and what does it still teach you?

LSH's promise was theoretical: a family of hash functions where `P[h(a) = h(b)]` is monotonically increasing in similarity, giving you sublinear query time *with a provable bound on the probability of missing a neighbour*. You hash every vector into `L` tables using `K` concatenated hash functions each, and at query time you look up the query's bucket in all `L` tables and exact-check the union of candidates. `K` controls precision (longer hashes → fewer, more-similar collisions), `L` controls recall (more tables → more chances to collide with the true neighbour). For cosine similarity, the classic construction is SimHash: random hyperplanes, `h(x) = sign(w · x)`, where `P[h(a) = h(b)] = 1 - θ(a,b)/π`.

It lost for one reason, and it is worth naming precisely because it generalises: **LSH is data-independent, and data-independent methods leave enormous performance on the floor when the data has structure.** Random hyperplanes know nothing about where your vectors actually are. IVF's centroids, PQ's codebooks, and HNSW's graph are all *fit to the corpus*. Empirically the gap is not marginal — at equal recall on standard benchmarks, graph methods run one to two orders of magnitude faster than tuned LSH, and LSH's memory blows up because you need many tables to reach high recall. The theoretical guarantee turned out to be worth less in practice than adapting to the data.

What it still teaches you, and why I would not skip it:

**One.** SimHash is the foundation of **binary quantization**, which is very much alive. Storing `sign(x)` as a bitstring and comparing with Hamming distance — `popcount(a XOR b)` — is the same idea, and at d=1536 it compresses 6144 bytes to 192 bytes (32×) with a distance computation that is 24 machine words of XOR-and-popcount. Modern schemes (RaBitQ) are essentially "SimHash done properly, with a rotation and an error bound."

**Two.** MinHash and its LSH banding are still the correct tool for **near-duplicate detection over sets** — deduping your corpus before you embed it, which is the highest-ROI thing you can do to a RAG index and which nobody does with vectors because Jaccard over shingles is both cheaper and more precise for that job.

**Three.** The `K`/`L` precision-recall decomposition is the cleanest available intuition for why every ANN method has exactly one recall knob that costs you candidates examined.

**🗣 Say this in the room:** "LSH lost because it is data-independent — random projections can't exploit the structure that learned centroids, codebooks and navigable graphs can, and empirically that's an order of magnitude at equal recall. I'd still reach for MinHash-LSH for corpus deduplication, and I'd note that binary quantization with Hamming rescoring is SimHash's direct descendant and is currently the fastest-moving corner of the field."

### Why did the field abandon methods with provable guarantees in favour of graph methods that have none?

Because the guarantee was over the wrong thing. LSH gives you a bound of the form "with probability ≥ 1−δ, you will find a point within `c` times the distance of the true nearest neighbour" — a *worst-case, data-independent, approximation-ratio* guarantee. Practitioners do not care about worst-case approximation ratio; they care about **average recall@10 on their actual query distribution at a fixed latency budget**. Those are different objectives and optimising the first does not optimise the second.

The intuition for why graphs win is worth being able to state cleanly, because it is the conceptual core of HNSW. A proximity graph over your vectors, searched greedily, exploits a property real embedding data has and random projections cannot see: **the local neighbourhood structure is navigable**. If I am at some vector and I look at its neighbours in the graph, at least one of them is usually closer to the query than I am. So greedy descent works, and it converges in roughly `O(log N)` hops if the graph has both short-range links (for the final approach) and long-range links (to cross the space quickly) — the small-world property. Random hyperplanes throw all of that away and start from scratch on every query.

The honest cost of the switch is exactly what an interviewer wants you to volunteer: **graph methods have no guarantee, no closed-form recall model, and pathological cases you discover empirically.** A graph built on clustered data with sparse inter-cluster links can trap greedy search in the wrong cluster. Deletes damage connectivity in ways no theory predicts. And recall depends on the *entry point* and the beam width in ways you can only measure. The discipline this imposes is the one I would enforce in review: **if you deploy a graph index you must also deploy the flat shadow evaluation from earlier in this section, because there is no theory that will tell you when it degrades.** With IVF you can at least reason about scanned fraction. With HNSW you measure or you fly blind.

That is also the modern synthesis: DiskANN's Vamana graph, ScaNN's quantization, and RaBitQ's bit codes all coexist because each attacks a different axis, and the winner is chosen by measurement on your corpus, not by theory. Anyone who tells you one index family dominates has not benchmarked on more than one dataset.
