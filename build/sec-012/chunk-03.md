### Longformer and BigBird both proposed sparse attention patterns years ago. Explain them, and explain why decoder-only LLMs didn't adopt them.

Both start from the same observation: the `T×T` attention matrix is mostly wasted, so hand-design a sparsity pattern that is `O(T)` and hope it preserves what matters.

**Longformer** uses three components: a sliding local window of width `w` around each token; *dilated* windows in some heads (attend to every `d`-th token within a wider span, exactly the dilated-convolution trick, which multiplies receptive field without multiplying cost); and a small set of **global tokens** that attend to everything and are attended by everything — in practice `[CLS]` for classification and all question tokens for QA. That last piece is the one that mattered: a global token is a designated all-to-all relay, so any two tokens are at most 2 hops apart.

**BigBird** uses window + global + **random**: each query also attends to `r` randomly chosen keys. The theoretical contribution is what made the paper famous — with random edges the attention graph becomes an expander, and they prove the resulting sparse transformer is still a universal approximator of sequence functions and is Turing complete, so sparsity does not cost you expressiveness *in principle*.

Why they did not take over decoder-only LLMs, in order of importance:

1. **They were encoder-era designs.** Both targeted BERT-style bidirectional encoders for classification and extractive QA, where "global tokens" is a natural concept because you know in advance which tokens matter (`[CLS]`, the question). In an autoregressive decoder there is no `[CLS]` and no question-known-in-advance; the whole point is that you do not know which token will turn out to matter.
2. **Random attention is a memory-access catastrophe.** Gathering `r` random KV vectors per query is uncoalesced HBM access. BigBird's implementation had to block-ify the randomness to be tractable at all, and even blocked random access defeats the contiguous tiling that FlashAttention depends on. The theory says sparsity is free; the memory system says otherwise.
3. **Fixed patterns are not learned.** The sparsity is a fixed structural prior chosen by a human before training. If the true dependency structure of your data does not match sliding+dilated+random, the model cannot route around it.
4. **The economics changed.** FlashAttention made *dense* attention IO-efficient enough that the constant-factor win from these patterns shrank dramatically, and GQA plus long-context training made dense 128k practical. The problem these papers solved got partially solved a different way.

**⚠ Trap:** citing BigBird's universal-approximation result as evidence that sparse attention is "free." The proof is about expressive capacity with unbounded depth and precision; it says nothing about whether gradient descent will *find* the function, or about wall-clock. A theoretically-sufficient architecture that runs at 30% MFU loses to a theoretically-redundant one that runs at 60%. This is the single most common way people over-read that paper.

### DeepSeek's NSA and Moonshot's MoBA revived sparse attention in 2025. What changed?

Two things changed, and they are exactly the two failure modes of the 2020 designs: the sparsity became **learned** rather than hand-designed, and it became **hardware-aligned** rather than pointwise-random.

**NSA (Native Sparse Attention)** combines three parallel branches whose outputs are blended by a learned gate: a **compression** branch that pools contiguous blocks of keys/values into coarse summary tokens so every query gets a cheap low-resolution view of the entire history; a **selection** branch that uses those compression scores to pick the top-`k` *blocks* of the original fine-grained KV and attends to them at full resolution; and a **sliding-window** branch for recent local context. The name's key word is "native" — the design is differentiable and trained from scratch with sparsity in place, rather than sparsifying a densely-trained model at inference. That distinction matters enormously: post-hoc sparsification always fights a model that learned to rely on the edges you deleted.

The hardware alignment is the other half. Selection operates on *blocks* whose size and layout are chosen so that a GQA group's queries all select the same blocks — which means the KV loads are contiguous and shared across the group, so the kernel gets coalesced reads and Tensor-Core-friendly tile shapes. That is a deliberate co-design of algorithm and memory system, and it is why NSA reports real end-to-end speedups rather than theoretical FLOP reductions.

**MoBA (Mixture of Block Attention)** takes the MoE framing literally: partition the KV sequence into blocks, and let each query token's router select the top-`k` blocks to attend to. Because it is structurally a gate over blocks, you can flip between full attention and sparse attention by changing `k`, which makes it practical to train mostly-dense and serve sparse, or to switch modes by layer.

**🗣 Say this in the room:** "The 2020 sparse-attention papers failed on two axes — the pattern was hand-designed rather than learned, and random gather is a memory-system disaster. NSA and MoBA fix both: the sparsity is a learned block-level gate, and the block granularity is chosen so that a whole GQA group loads the same contiguous KV tiles. It's algorithm-hardware co-design, not just a FLOP-count argument."

**⚠ Trap:** presenting these as drop-in inference optimizations. They are *pretraining architectures*. You cannot take a dense Llama checkpoint, bolt on NSA, and get NSA's numbers — the model never learned to route through a gate. There are post-hoc KV-sparsification methods (H2O-style heavy-hitter eviction, quest-style block selection at decode) that *are* drop-in, but they are a different family with a different quality profile, and conflating them is a tell.

### Explain Ring Attention. What problem does it solve that tensor parallelism doesn't?

The problem: at 1M tokens, the *activations for a single sequence* do not fit on one device, and neither does the KV. Tensor parallelism shards the *model* — each device holds a slice of every weight matrix — but every device still processes the full sequence, so activation memory per device is unchanged. You need to shard the **sequence dimension** instead, and the obstacle is that attention is all-to-all across that dimension: query block `i` needs key block `j` for every `j ≤ i`.

Ring Attention's mechanism rests on the same property FlashAttention rests on: attention can be computed blockwise with an **online softmax**, accumulating a running max and a running normalizer, so you never need all keys simultaneously. So: put device `d` in charge of sequence block `d`, holding its own Q, K, V. Each device computes attention of its local Q against its local KV, updating running softmax statistics. Then every device sends its KV block to its right neighbour in a ring and receives one from its left. Compute again. After `P` steps, every Q block has seen every KV block, and the online-softmax accumulators are exactly correct.

The reason this is not just "distributed attention" but a genuinely good idea is the overlap argument. Compute for one block-pair is `O(c² · d)` where `c` is block size; the communication of one KV block is `O(c · d)` bytes. So the compute-to-communication ratio grows linearly in `c`. **Pick `c` large enough and the ring transfer is entirely hidden behind the previous block's matmul — the communication becomes free.** That is the whole trick, and it means context length scales with device count at near-zero communication overhead, hence the paper's "near-infinite context" claim.

**📄 Paper:** Liu, Zaharia & Abbeel (2023), *Ring Attention with Blockwise Transformers for Near-Infinite Context* — sharded the sequence dimension across devices with a ring KV rotation overlapped with blockwise attention compute, replacing "one sequence must fit on one device" as a hard constraint.

**⚠ Trap:** the causal load imbalance, which is the follow-up question. With a naive contiguous split, device 0 holds the first block and its queries attend to almost nothing; the last device's queries attend to everything. Under causal masking device `P−1` does roughly `P×` the work of device 0, and the ring runs at the speed of its slowest member, so you waste close to half your cluster. The fix is to *stripe* the assignment — give each device a strided subset of positions rather than a contiguous block — so each device gets a mix of early and late queries and the causal work is balanced. This is Striped Attention, and knowing that the naive version has ~50% waste is exactly the kind of detail that separates "I read the abstract" from "I would deploy this."

### Derive linear attention. What exactly do you lose when you drop the softmax?

Write standard attention without the softmax notation, as a normalized weighted sum:

```
out_i = Σ_j sim(q_i, k_j) · v_j  /  Σ_j sim(q_i, k_j),      sim(q,k) = exp(q·k / √d)
```

The `T²` cost comes entirely from `exp` being non-factorizable: you cannot pull `q_i` out of the sum because `exp(q·k)` does not decompose into a product of a `q`-only term and a `k`-only term. So replace it with something that *does*: pick a feature map `φ` and set `sim(q,k) = φ(q)·φ(k)`. Now:

```
numerator_i   = Σ_j (φ(q_i)^T φ(k_j)) v_j = φ(q_i)^T · [ Σ_j φ(k_j) v_j^T ]
denominator_i = φ(q_i)^T · [ Σ_j φ(k_j) ]
```

The bracketed terms do not depend on `i`. Compute them once: `S = Σ_j φ(k_j) v_j^T`, a `[d_φ, d_v]` matrix, and `z = Σ_j φ(k_j)`, a `[d_φ]` vector. Total cost `O(T · d_φ · d_v)` — linear in `T`. Under causal masking, `S` and `z` become *running* sums updated one token at a time, which means **linear attention with a causal mask is literally an RNN with a matrix-valued hidden state**, decoding in `O(1)` time and `O(d_φ · d_v)` memory per step with no KV cache at all.

```python
# causal linear attention, one head. q,k,v: [T, d]
def linear_attn(q, k, v, phi=lambda x: torch.nn.functional.elu(x) + 1):
    T, d = q.shape
    S = torch.zeros(d, v.shape[-1]); z = torch.zeros(d); out = []
    for t in range(T):                      # sequential form == the RNN view
        kt = phi(k[t])
        S = S + torch.outer(kt, v[t])       # [d, d_v] running state
        z = z + kt
        qt = phi(q[t])
        out.append((qt @ S) / (qt @ z + 1e-6))
    return torch.stack(out)
```

What you lose is **selectivity**, and the argument is a counting argument that I want you to be able to give crisply. Softmax attention's "state" at position `T` is the full KV cache: `T × d` numbers, growing without bound. Linear attention's state is `d_φ × d_v` numbers, *fixed*. With `d = 64`, that is 4,096 numbers no matter whether `T` is 1,000 or 1,000,000. At `T = 100,000` the softmax model is carrying 6.4M numbers per head and the linear model 4,096 — a **1,560× compression**. Information-theoretically, you cannot losslessly recall an arbitrary one of 100,000 distinct items from 4,096 numbers, so exact recall must degrade. And the failure is not graceful noise: because every new token adds a rank-1 outer product to `S`, old information is *overwritten by interference*, not forgotten cleanly.

Softmax's exponential is what makes it *selective* — it can put nearly all mass on one key, which is what a lookup requires. A bounded-rank linear map fundamentally averages.

**📄 Paper:** Katharopoulos et al. (2020), *Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention* — established the associativity trick and the RNN equivalence with `φ(x) = elu(x)+1`. Choromanski et al. (2021), *Rethinking Attention with Performers*, gave the FAVOR+ random-feature map that *approximates* the softmax kernel unbiasedly rather than replacing it, which is the more principled variant — but the approximation variance grows with sequence length and the fixed-state recall ceiling is the same.

**🗣 Say this in the room:** "Linear attention is the associativity trick: drop `exp`, and `(QK^T)V` becomes `Q(K^TV)`, which is linear in `T` and, under causal masking, is exactly an RNN with a `d×d` matrix state. What you buy is `O(1)` decode with no KV cache. What you pay is that your entire history is compressed into a fixed-size state — at 100k tokens that's a 1,500× compression versus a KV cache, so exact single-fact recall has to degrade. It's a great trade for gist and a terrible one for lookup."

### Teach me state space models from scratch. Start with S4 and the recurrence-convolution duality.

Start with a continuous linear dynamical system, which is the same object a control engineer would write down:

```
h'(t) = A h(t) + B x(t)          # state evolves
y(t)  = C h(t) + D x(t)          # output reads the state
```

Discretize with step `Δ` and you get a linear recurrence: `h_t = Ā h_{t−1} + B̄ x_t`, `y_t = C h_t`. That is an RNN, and it decodes in `O(1)` per token with a fixed state of size `N` — this is the inference-time picture and it is why SSMs are attractive at all.

The magic is the training-time picture. Because the recurrence is *linear* and *time-invariant* (`Ā`, `B̄`, `C` do not depend on `t` or on `x`), you can unroll it in closed form:

```
y_t = Σ_{k=0..t} (C Ā^k B̄) · x_{t−k}
```

which is a **convolution** of the input with a kernel `K = (CB̄, CĀB̄, CĀ²B̄, ...)`. So the same layer has two exact, mathematically identical forms: a sequential recurrence for `O(1)` decode, and a length-`T` convolution that can be computed for the entire sequence in parallel via FFT in `O(T log T)`. **Train as a convolution, serve as a recurrence.** That duality is the entire reason SSMs are competitive — an RNN you can train with full sequence parallelism.

S4's specific contribution was making this actually work. A randomly-initialized `A` gives you a kernel that either explodes or decays to nothing within a few steps, so the model has no long memory. S4 initializes `A` with **HiPPO** structure — a matrix derived from the theory of optimally projecting a function's history onto an orthogonal polynomial basis, so the state provably maintains a compressed reconstruction of the entire past. It also uses a diagonal-plus-low-rank parameterization so that computing `Ā^k` is tractable. The result was state-of-the-art on Long Range Arena by an enormous margin, on tasks with 16k-length dependencies where transformers were failing.

**📄 Paper:** Gu, Goel & Ré (2022), *Efficiently Modeling Long Sequences with Structured State Spaces (S4)* — made deep linear SSMs trainable at scale via HiPPO initialization and a structured parameterization of `A`, replacing gated RNNs as the long-range-dependency architecture on synthetic benchmarks.

**⚠ Trap:** describing an SSM as "a fast approximation to attention." It is not an approximation to anything. It is a different function class — a linear time-invariant filter bank with a learned decay spectrum. It has no notion of comparing a query to a key. Getting this wrong makes every subsequent answer about recall behavior incoherent.

### S4 was linear time-invariant. Mamba isn't. What did selectivity fix, and what did it cost?

The defect in S4 is the "time-invariant" part. `Ā`, `B̄`, `C` are the same at every position, so the layer is a fixed filter — it applies the same decay and the same read-out whether the incoming token is "the" or a critical account number. It cannot decide *what to remember*. The diagnostic test is selective copying: given a stream with a few important tokens scattered among noise, reproduce only the important ones. S4 fails it, because filtering by content is exactly what a content-independent filter cannot do.

Mamba's fix: make `Δ`, `B`, and `C` **functions of the input token**. `Δ_t = softplus(Linear(x_t))` and similarly for `B_t`, `C_t`. Now `Δ_t` is a per-token, per-channel gate on how much the state updates — large `Δ` means "this token matters, write it in and decay the old state"; near-zero `Δ` means "ignore this token, hold state." That is a content-based gate, and it recovers selective copying and induction-head-style behavior.

The cost is that you just destroyed the convolution. `Ā_t` now varies with `t`, so `y_t = Σ (C Ā^k B̄) x_{t−k}` is no longer a convolution and there is no FFT path. Mamba's answer is a **hardware-aware parallel scan**: the recurrence is still *associative* (composition of affine maps is an affine map), so a Blelloch-style parallel scan computes it in `O(log T)` depth. The implementation keeps the expanded state in SRAM rather than materializing it to HBM — the state is `batch × d_inner × N`, which at `d_inner` 8192 and `N` 16 is large enough that writing it out per timestep would make the layer entirely bandwidth-bound — and recomputes it in the backward pass instead of storing it, exactly the FlashAttention playbook applied to a scan.

**📄 Paper:** Gu & Dao (2023), *Mamba: Linear-Time Sequence Modeling with Selective State Spaces* — made SSM parameters input-dependent, trading the convolutional training path for a hardware-aware associative scan, and was the first SSM to be competitive with transformers on general language modeling.

**📐 Numbers you must know — the state-size comparison, which is the whole story.** A Mamba layer's recurrent state is `d_inner × N` numbers per sequence. With `d_model = 4096`, expansion 2 (`d_inner = 8192`) and `N = 16`, that is **131,072 numbers per layer, constant in sequence length**. A comparable GQA transformer layer with 8 KV heads and `d_head` 128 caches `2 × 8 × 128 = 2,048` numbers *per token*. At `T = 100,000` that is **204.8M numbers per layer**. The SSM state is ~1,560× smaller — and *that ratio is the entire recall trade-off*, expressed as a number. Everything about SSM behavior on retrieval tasks follows from it.

### What is Mamba-2's state space duality, and why does it matter practically?

Mamba's scan is elegant but it is not a matmul, so it does not use Tensor Cores, which means it runs at a small fraction of the FLOP/s the hardware can deliver. That is the practical problem Mamba-2 solves.

The theoretical framing: restrict `A` from a diagonal matrix to a **scalar times identity** per head. That looks like a loss of expressiveness, but it makes the recurrence's unrolled form expressible as a structured matrix — specifically a **semiseparable** matrix — and multiplying by a semiseparable matrix decomposes into block matmuls. So the same layer can be computed either as a linear-time scan or as a quadratic-time masked matmul, and the paper shows the quadratic form is *exactly* a masked linear attention. Hence "state space duality": SSMs and a family of linear attention variants are two views of the same object, and the algorithm can pick whichever view is faster for the current shape.

Why it matters practically, in two concrete consequences. First, **speed**: the chunked matmul form uses Tensor Cores, and Mamba-2 reports substantially faster training than Mamba-1 at equal quality. Second, and more importantly for the recall discussion, **the state got bigger**. Because the constrained `A` makes the compute matmul-shaped, you can afford a much larger `N` — Mamba-2 uses state dimensions in the 64–256 range versus Mamba-1's 16. Going from `N = 16` to `N = 256` is a **16× larger recurrent state**, which directly buys back recall capacity. That is why essentially every serious 2025-era hybrid uses Mamba-2 blocks rather than Mamba-1.

**📄 Paper:** Dao & Gu (2024), *Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality* — unified SSMs with masked linear attention through semiseparable matrices, enabling a matmul-based chunked algorithm and much larger state dimensions.

**🗣 Say this in the room:** "Mamba-2's contribution is that a scalar-times-identity `A` makes the unrolled recurrence a semiseparable matrix, which factors into block matmuls — so the scan becomes Tensor-Core work and is dual to masked linear attention. The practical payoff isn't just speed: it lets you afford a 16× larger recurrent state, which is the thing that was actually limiting recall."

### Why do pure SSMs fail needle-in-a-haystack? Give me the mechanism, not the observation.

The mechanism is a fixed-capacity channel with destructive interference, and I want to give it in three steps.

**Step one — the capacity bound.** An SSM compresses the entire history into a state of fixed size `S` (for Mamba-2, `d_inner × N` numbers per layer). A transformer keeps `T × 2 × n_kv_heads × d_head` numbers, which *grows*. Needle-in-a-haystack requires reproducing an arbitrary token span chosen adversarially from anywhere in a 100k-token context. If the needle can be any of ~100k positions with arbitrary content, the information required to answer is not compressible in advance — the model does not know at write time which token will be queried. A fixed-size state must therefore lossily compress, and the compression must be lossy on *something*.

**Step two — what gets compressed away.** The selection gate `Δ_t` decides write strength from the *token itself*, at the moment it arrives, with no knowledge of the future question. A random 8-digit code embedded in a legal document has no lexical property marking it as important; the gate has no basis to write it strongly. So the model learns a heuristic — write surprising or rare tokens strongly — which correlates with importance but is not it. A needle that is lexically ordinary is filtered out at write time and no amount of decoding can recover it.

**Step three — interference, not decay.** Each update adds a term to `S`. Later tokens do not simply push old ones off the end of a queue; they superpose onto the same finite-dimensional state. Reading out an old item means projecting `S` onto a direction and getting the true value plus a sum of cross-terms from every intervening token. As `T` grows, the interference noise grows relative to the signal, so recall degrades continuously rather than cliff-edging. That is why SSM needle curves slope down smoothly with distance instead of failing abruptly — a very recognizable signature.

Contrast attention: a query is compared against every stored key, and the softmax can place ~all mass on one. Attention is *content-addressable memory with lookup at read time*. An SSM is *lossy compression with the decision made at write time*. Retrieval is a read-time-decided problem, which is why the architecture that decides at read time wins it.

**🗣 Say this in the room:** "Because an SSM decides what to keep at write time and attention decides what to retrieve at read time. The gate has to judge a token's importance the moment it arrives, before it knows the question — so a lexically-unremarkable needle gets a small write, and everything after it superposes onto the same fixed-size state. That's interference, not forgetting, which is why the recall curve slopes down smoothly instead of falling off a cliff."

**⚠ Trap:** concluding "SSMs are bad." They are *not* bad at language modeling — perplexity on natural text is competitive, because natural text is dominated by local and diffuse dependencies that a fixed state handles fine. Perplexity is a terrible predictor of retrieval, and this is the cleanest example of that in the whole field. If your eval suite is perplexity plus a few short benchmarks, an SSM will look excellent and then fail catastrophically the first time a user pastes a document and asks a question about paragraph 40.

### So why do hybrid stacks work? Walk me through Jamba, Nemotron-H, Falcon-H1 and how you'd choose a ratio.

Hybrids work because the two failure modes are complementary and the costs are asymmetric. Attention gives content-addressable read-time lookup at `O(T)` cache; SSM gives `O(1)` state at the cost of read-time lookup. You do not need *every* layer to be content-addressable — you need enough attention layers to serve as long-range lookup, and the rest of the depth can be cheap SSM work. Because KV cost is a *sum over layers*, replacing 7 of every 8 layers with Mamba cuts KV cache by ~8× while leaving genuine attention lookup available.

**Jamba** (AI21, 2024) was the first at real scale: blocks of 8 layers with a 1:7 attention-to-Mamba ratio, plus MoE on alternating layers. The headline claim was that this made very long context fit on far less HBM than a comparable transformer, which is exactly the KV-sum argument.

**Nemotron-H** (NVIDIA, 2025) pushed the ratio further — the great majority of self-attention layers replaced with Mamba-2, with only a small single-digit percentage remaining as attention — reporting comparable accuracy at meaningfully higher inference throughput.

**Falcon-H1** (TII, 2025) took a different structural bet: rather than *interleaving* attention and SSM layers, run attention heads and Mamba-2 heads **in parallel within the same block** and combine their outputs. The argument is that the two mechanisms are complementary at every depth rather than at alternating depths, and it decouples the "how much attention" knob from the layer count.

(**📅 Volatile — see §5:** exact ratios, layer counts and throughput claims move with every release; read the config, do not recite.)

**How I would choose the ratio.** Not by perplexity — perplexity is nearly flat across a wide range of ratios and will mislead you. I would sweep the ratio and plot **multi-hop needle accuracy at the target context length** against **KV bytes per token**. The curve has a knee: below some number of attention layers, multi-hop recall falls off sharply because a two-hop lookup needs at least two attention layers positioned after the relevant information has been assembled in the residual stream. Above the knee, adding attention layers buys little accuracy and costs linear KV. Pick the knee plus one layer of margin.

Two structural details I would insist on: put at least one attention layer reasonably **early** (so long-range information can enter the stream before the SSM layers compress it) and at least one **late** (so the final lookup can be performed against a well-formed representation). A stack with all its attention layers clustered in the middle underperforms an evenly-spread one at the same ratio.

**💰 Math:** a 1:7 hybrid at 128k context. Transformer baseline at 128 KiB/token gives 131,072 × 128 KiB = **16 GiB per sequence**. Hybrid at 1/8 the attention layers gives **2 GiB** (the SSM state is a fixed ~100 MB-scale term, negligible by comparison). On an 80 GB card with ~40 GiB free after weights, that is 40/16 = **2.5** concurrent 128k sessions versus 40/2 = **20** — **8× the concurrency on identical hardware**. For a 1,000-concurrent-session product that is 400 GPUs versus 50; at $2.50/GPU-hour × 730 h that is **$730k/month versus $91k/month**. This is why hybrids are not an academic curiosity.

### Where does RWKV fit in this landscape?

RWKV is the linear-attention/RNN branch that got to competitive language-model scale before Mamba did, developed largely in the open by a community effort rather than a lab. The architecture keeps a transformer-like block structure — a token-mixing sublayer and a channel-mixing sublayer with residuals and norms — but replaces attention with a "WKV" operator: a weighted sum over history where the weights are an exponentially-decaying function of distance, with a learned per-channel decay rate and a special bonus term for the current token. Because the decay is exponential and per-channel, the whole thing admits a recurrent formulation with fixed state and can be trained with a parallel formulation over the sequence.

Its position in the taxonomy: same fixed-state family as linear attention and SSMs, same recall ceiling, same `O(1)` decode and no KV cache. Later versions (Eagle/Finch/Goose generations) added data-dependent decay — which is structurally the same move Mamba made with selectivity, arrived at independently — and a matrix-valued state with a delta-rule-style update that improves in-context recall.

What I would actually say about it in an interview: RWKV matters as evidence that the fixed-state family is a real architecture family and not a one-lab phenomenon, and it matters practically for edge and CPU deployment, where having no KV cache is worth more than exact recall. If someone asks me to deploy a 7B-class model on a device with 4 GB of RAM and unbounded conversation length, RWKV-class models are a genuinely serious option in a way a transformer is not. For a long-document QA product they are not.

**⚠ Trap:** describing RWKV as "an RNN, so it must be slow to train." The entire point is that the linear-with-exponential-decay structure gives a parallel training form, exactly as with linear attention and S4. Sequential training is the property of *nonlinear* recurrences (LSTM, GRU); it is not a property of recurrence per se.

### I'm serving a hybrid Mamba-attention model. What changes about my inference stack?

More than people expect, and this is a good question because it is where architecture meets the thing you already know how to build.

**The cache is no longer just KV.** You now have two kinds of per-sequence state: paged KV blocks for the attention layers, and a **dense, fixed-size recurrent state tensor** for each SSM layer. The SSM state does not grow with sequence length, which is the good news, and it cannot be paged in the PagedAttention sense, which is the bad news — it is one contiguous `[d_inner, N]` tensor per layer per sequence that must be allocated in full at admission time. So your memory model becomes `fixed_per_sequence_ssm_state + variable_paged_kv`. Admission control has to account for both, and the fixed term means the *minimum* footprint of a sequence is higher than a transformer's even though its maximum is far lower.

**Prefix caching gets harder, and this is the big one.** Transformer prefix caching works because KV blocks are a *function of the prefix alone* and are position-addressable — you can keep block 3 of a prefix and reuse it. The SSM state is a *running summary*: to reuse a shared prefix you must snapshot the recurrent state at the end of that prefix and restore it, and you cannot reuse a *suffix* or splice in the middle, because state at position `t` depends on every token before it. So SSM prefix caching means checkpointing dense state tensors keyed on the prefix hash, with a very different memory profile — fixed-size, non-shareable-by-block entries rather than a block pool with copy-on-write. Several engines support this now; verify it explicitly rather than assuming.

**Chunked prefill and continuous batching change shape.** The scan is sequential in `t`, so a chunked prefill must carry state across chunks in order — you cannot process chunk 5 before chunk 4 the way you can with attention. Batching sequences at different positions is fine (each has its own state), but a scan kernel batched over sequences with wildly different chunk sizes has ragged-shape problems that attention kernels solved years ago and scan kernels are still catching up on.

**Speculative decoding still works** and is arguably more attractive, since verification of `k` draft tokens is a `k`-step scan rather than a `k`-token attention pass — but rollback is subtler: rejecting a draft token means restoring the SSM state to a prior step, so you must snapshot state at the verification boundary rather than just truncating a cache.

**🔍 Failure taxonomy — hybrid serving.** (1) Throughput far below the paper's claim → check whether the engine has a fused chunked-scan kernel or is falling back to a sequential reference implementation; the gap is often 5–10×. (2) Memory OOM at low concurrency despite tiny KV → the fixed SSM state per sequence dominates; compute `n_ssm_layers × d_inner × N × dtype_bytes` and multiply by max concurrency. (3) Cache-hit rate near zero on a workload with a big shared system prompt → the engine is caching KV but not snapshotting SSM state, so every request re-scans the prefix. (4) Quality fine at batch 1, degraded at high batch → ragged-batch scan kernel bug; test with a fixed-length batch to isolate.

### Code assistant over a 200k-line repo. 128k-context transformer or a hybrid SSM? Pick, and tell me what would change your mind.

I take the 128k transformer, and the reason is that the workload is *retrieval-shaped*, which is the one thing the fixed-state family is structurally bad at.

The core operation in a code assistant is: the user is editing `handlers/billing.py` line 400, and correctness requires the exact signature of a function defined in `core/models.py`, the exact name of an enum member declared in `constants.py`, and the exact type of a column in a migration. These are single-occurrence, lexically-unremarkable tokens whose importance is determined by the *question*, not by the token. That is precisely the read-time-lookup problem, and an SSM's gate must decide at write time whether `MAX_RETRY_BACKOFF_SECONDS = 30` is worth writing strongly, with no idea that the user will ask about it 90k tokens later. The failure mode is not "slightly worse code" — it is a confidently hallucinated signature, which in a code assistant is worse than no answer because it costs the user a debugging cycle.

Second argument: prefix caching economics dominate here and favour the transformer. A code assistant re-sends a large, mostly-stable context on every keystroke-triggered completion or every chat turn. Transformer prefix caching on a 100k-token repo context turns a 100k prefill into a few-hundred-token prefill on the hit path. That is the difference between a 4-second and a 200-millisecond TTFT, and TTFT is the metric this product lives or dies on. SSM prefix caching exists but is less mature and cannot do partial/block-level reuse, so a small edit high in the context invalidates everything after it — which, in an editor, is the common case.

What would change my mind, stated as explicit triggers:

- **If the hybrid ratio keeps enough attention layers that multi-hop needle accuracy on my own repo corpus is within 2 points of the transformer at 128k**, the 8–10× KV reduction becomes very hard to argue against, and I would switch. That is an empirical question I would settle with the ratio sweep rather than an architectural prejudice.
- **If the product moves to unbounded-length streaming sessions** — a long-running agent that accumulates hours of tool output — the `O(1)` state stops being a nice-to-have and becomes the only way to bound memory.
- **If I am deploying on-device** (local code completion on a laptop with 16 GB unified memory), no KV cache wins outright and I accept the recall cost by pairing it with an explicit retrieval tool.

And the answer I would actually ship regardless of architecture: **do not rely on raw long context for symbol lookup at all.** Build a real code index — an LSP or tree-sitter symbol table — and retrieve exact definitions into the prompt. That converts a hard associative-recall problem into a cheap deterministic one, costs a few thousand tokens instead of a hundred thousand, and works on both architectures. Long context is the fallback for the cases the index misses, not the primary mechanism.

**🗣 Say this in the room:** "Transformer, because a code assistant is a read-time exact-lookup workload and fixed-state models decide what to keep at write time. But the real answer is that I would not lean on long context for symbol resolution at all — a tree-sitter or LSP index retrieving exact definitions is cheaper, deterministic, and architecture-independent. Long context is the fallback, not the mechanism."
