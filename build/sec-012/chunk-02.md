### Derive Position Interpolation for me. Why does squeezing positions work better than letting them run off the end?

The insight is a one-line reframing that is obvious in hindsight and was not obvious in 2023: instead of asking the model to evaluate rotation angles it has never seen, ask it to evaluate angles it has seen *at finer granularity*. Interpolation is in-distribution; extrapolation is not.

Mechanically, Position Interpolation replaces position `m` with `m/s`, where `s = target_length / original_length`. Equivalently — and this is how it is implemented — you divide every inverse frequency by `s`, so `θ_i → θ_i/s`. Extending a 4k model to 32k means `s = 8`, and position 32,000 now produces the rotation angles that position 4,000 produced during training. Every angle the model sees at inference lies inside the arc it was trained on. Positions 0 and 1, which used to be `θ_i` apart, are now `θ_i/8` apart.

That last sentence is also the cost. You have compressed the angular resolution between adjacent tokens by 8×. In the high-frequency dimensions — the ones responsible for "is this the previous token or the one before it" — you have made neighbours 8× harder to distinguish. Chen et al. found this is recoverable: 1,000 fine-tuning steps on long sequences was enough to restore quality at the extended length, which is a trivially small budget compared to pretraining. But it is *not* free zero-shot, and it measurably degrades short-context performance, because a model with PI applied is now using compressed angles even for a 200-token prompt.

**📄 Paper:** Chen et al. (2023, Meta), *Extending Context Window of Large Language Models via Positional Interpolation* — replaced naive extrapolation with linear position downscaling plus a short fine-tune, and showed 2k→32k extension with about 1,000 steps. Everything since (NTK-aware, YaRN, LongRoPE, Llama-3.1's scaling) is a refinement of "which frequencies do you interpolate and by how much," not a departure from the idea.

**⚠ Trap:** applying PI and only measuring at the *new* long length. PI trades short-context accuracy for long-context viability, uniformly across all frequencies. Your 128k eval improves and your 2k eval quietly regresses, and 95% of your production traffic is under 2k. **The rule I enforce: any context-extension change must report a short-context regression number alongside the long-context gain.** If nobody measured the 512-token case, the change is not reviewed.

### Linear PI hurts short-range resolution. What's the NTK-aware fix and what's the intuition?

The intuition is that PI applies the same medicine to patients with completely different diseases. RoPE's frequency spectrum spans wavelengths from ~6 tokens to ~54k tokens (base 10,000, `d_head` 128). The *high-frequency* dimensions with wavelength 6 or 20 tokens already wrapped around thousands of times during training on 4k sequences — the model has seen every possible angle in those subspaces. They need no help extending; they are already fully in-distribution at any length. The *low-frequency* dimensions with wavelength 54k tokens only ever swept a 7% arc during 4k training — those are the ones that go out of distribution. PI penalizes both equally, destroying resolution in the dimensions that never needed fixing.

NTK-aware scaling (originally posted by the community researcher "bloc97" as *NTK-aware scaled RoPE*, later formalized inside YaRN) does the frequency-dependent thing: instead of dividing all frequencies by `s`, it *raises the base* by a factor chosen so that the highest frequency is essentially untouched and the lowest frequency is interpolated by roughly the full factor `s`. Concretely `base' = base · s^(d/(d−2))`. Because `θ_i = base^(−2i/d)`, raising the base scales frequency `i` by `s^(−2i/(d−2))` — which is ≈1 at `i=0` (no change to the fastest wheel) and ≈`1/s` at `i = d/2−1` (full interpolation of the slowest wheel), with a geometric ramp in between.

The name comes from a neural-tangent-kernel argument about networks struggling to learn high-frequency features, which is honestly the weakest part of the story; the empirical result is what matters and it is solid. NTK-aware scaling gives usable zero-shot extension — 2× to 4× without any fine-tuning at all, where PI zero-shot is noticeably degraded — and better results than PI after fine-tuning too.

**⚠ Trap:** conflating "NTK-aware scaling" with "just raise `rope_theta`." They are the same *operation* but different *intents* and different magnitudes. Llama-3 raising the base from 10,000 to 500,000 was done during pretraining as an architectural choice. NTK-aware scaling computes a specific base multiplier from the extension factor `s` and applies it to an already-trained checkpoint. Applying an arbitrary base to a trained checkpoint without the `s`-derived formula is guesswork, and applying it without any fine-tune degrades short context.

### Walk me through YaRN in enough detail that I could implement it. What are its two independent contributions?

YaRN's mental model: NTK-aware got the *direction* right (treat frequencies differently) but applies a smooth geometric ramp across all dimensions, when what you actually want is a hard three-way classification with a small transition band. And separately, any interpolation scheme dilutes attention entropy at long context, which needs its own fix. So YaRN has two independent pieces, and I would present them as two because interviewers frequently only know the first.

**Piece one: NTK-by-parts.** Classify each frequency dimension by comparing its wavelength `λ_i = 2π/θ_i` to the *original* training context `L`:

- If `λ_i` is much shorter than `L` (the model saw many full rotations in this subspace during training), leave the frequency **completely untouched** — pure extrapolation. These are the fine-grained local-order dimensions.
- If `λ_i` is longer than `L` (the model never completed a rotation), apply **full interpolation**: `θ_i → θ_i/s`.
- In between, **linearly ramp** between the two using a smoothing factor, to avoid a discontinuity in the spectrum.

The thresholds are set by two hyperparameters `α` and `β` (the paper uses α=1, β=32 for Llama models), where the boundaries are at `L/β` and `L/α` in wavelength terms — equivalently, "did this dimension complete at least β rotations" and "at least α rotations."

**Piece two: attention temperature.** When you interpolate positions, the distribution of `q·k` scores changes — the effective spread of relative-position contributions shrinks and attention gets flatter, which hurts at long context. YaRN adds a constant temperature on the attention logits, `softmax(q·k / (t·√d))`, and empirically fits `√(1/t) = 0.1·ln(s) + 1` for extension factor `s`. The beautiful part is the implementation: since `t` is a *constant*, you fold `√(1/t)` into the precomputed cos and sin tables. The attention kernel is unmodified, the cost is zero, and it is why YaRN is a drop-in change to the rotary embedding module and nothing else.

Together they get YaRN to what the paper reports as roughly 10× extension with about 0.1% of the original pretraining tokens of fine-tuning — dramatically cheaper than PI's requirements at large `s`, and with usable zero-shot behavior too.

**📄 Paper:** Peng, Quesnelle, Fan & Shippole (2023), *YaRN: Efficient Context Window Extension of Large Language Models* — combined per-wavelength selective interpolation with a logarithmic attention-temperature correction, replacing uniform PI and pure NTK-aware base scaling as the default extension recipe.

**🗣 Say this in the room:** "YaRN is two things. NTK-by-parts: leave high-frequency dimensions alone because they already wrapped many times in training, fully interpolate low-frequency ones that never completed a rotation, and ramp between. Plus an attention temperature that grows like `0.1·ln(s) + 1` to counteract entropy dilution — folded into the cos/sin table, so it costs nothing at inference."

### What does LongRoPE add on top of YaRN?

LongRoPE's premise is that YaRN's three-zone classification with a hand-chosen α and β is a coarse, human-designed approximation to a per-dimension rescaling that could just be *searched*. So instead of a formula, LongRoPE runs an evolutionary search over a rescale factor for each of the `d_head/2` frequency dimensions independently, using perplexity on long sequences as the fitness function. It also searches over the number of initial tokens to leave un-rescaled — an explicit nod to attention sinks, where the first few tokens behave differently enough that a global rescale hurts them.

The second contribution is progressive extension: search a rescaling to get to a medium length, fine-tune there, then search again from the fine-tuned checkpoint to reach the final length, rather than attempting one enormous jump. This is the same principle as a length curriculum and it is why the paper can report reaching multi-million-token context from a 4k/8k base.

The third contribution, and the one most relevant to production, is that LongRoPE keeps a *separate* rescaling configuration for short sequences and switches between them based on input length. That directly addresses the trap from the PI question: rather than accepting a short-context regression, you run the unscaled (or lightly scaled) rotary config when the input is short and the extended config when it is long.

**📄 Paper:** Ding et al. (2024, Microsoft), *LongRoPE: Extending LLM Context Window Beyond 2 Million Tokens* — replaced hand-derived per-frequency scaling formulas with an evolutionary search over per-dimension rescale factors plus progressive extension and a dual short/long configuration.

**⚠ Trap:** the dual-config trick is the part people miss, and it has a nasty serving consequence. If your rotary configuration depends on input length, then **two requests with the same prefix but different total lengths produce different K vectors for that prefix**, and your prefix cache hit rate collapses at the length boundary. Any length-conditional positional scheme needs the cache key to include the config bucket, or you will serve corrupted KV. Whenever someone proposes length-conditional anything in the positional path, my first question is "what happens to the prefix cache."

### Here's a Llama-3.1 `rope_scaling` block. Read it to me and tell me exactly what happens at inference.

```json
"rope_theta": 500000.0,
"max_position_embeddings": 131072,
"rope_scaling": {
    "rope_type": "llama3",
    "factor": 8.0,
    "low_freq_factor": 1.0,
    "high_freq_factor": 4.0,
    "original_max_position_embeddings": 8192
}
```

This is a YaRN-family NTK-by-parts scheme with Meta's own parameterization. Read it as follows. The base model was trained at 8,192 context with RoPE base 500,000. `factor: 8.0` means the target is 8 × 8,192 = 65,536 — note that this is *less* than the advertised 131,072, which is your first clue that the advertised number involves training beyond the pure scaling factor. The two `*_freq_factor` fields define wavelength thresholds in units of the original context: `low_freq_wavelen = 8192/1.0 = 8192` tokens and `high_freq_wavelen = 8192/4.0 = 2048` tokens.

Then, per frequency dimension `i` with wavelength `λ_i = 2π/θ_i`:

- `λ_i < 2048` → **untouched**. Extrapolate. This dimension completed at least 4 full rotations in the original 8k window.
- `λ_i > 8192` → **fully interpolated**, `θ_i → θ_i/8`. This dimension never completed even one rotation in 8k.
- `2048 ≤ λ_i ≤ 8192` → **linear ramp** between the two, with smoothing factor `(8192/λ_i − 1.0)/(4.0 − 1.0)`.

Now compute where the boundaries actually land, because that is the answer an interviewer is fishing for. With `d_head = 128` there are 64 frequency pairs and `λ_i = 2π · 500000^(i/64)`. Setting `λ = 2048`: `500000^(i/64) = 2048/2π = 325.9`, so `i/64 = ln(325.9)/ln(500000) = 5.787/13.122 = 0.441`, giving **i ≈ 28**. Setting `λ = 8192`: `8192/2π = 1303.8`, `i/64 = 7.173/13.122 = 0.547`, giving **i ≈ 35**.

So concretely: **frequency pairs 0–28 are untouched, pairs 29–35 ramp, pairs 36–63 are divided by 8.** Roughly 45% of the spectrum is left alone, ~11% transitions, ~44% is fully interpolated. That is the whole config in one sentence, and being able to produce those indices from the JSON is the difference between "I've read the docs" and "I've debugged this."

```python
import math, torch

def llama3_inv_freq(d_head=128, base=500000.0, factor=8.0,
                    low_freq_factor=1.0, high_freq_factor=4.0, orig_ctx=8192):
    inv = 1.0 / (base ** (torch.arange(0, d_head, 2).float() / d_head))
    wavelen = 2 * math.pi / inv
    low_wl, high_wl = orig_ctx / low_freq_factor, orig_ctx / high_freq_factor
    inv_interp = inv / factor
    smooth = (orig_ctx / wavelen - low_freq_factor) / (high_freq_factor - low_freq_factor)
    inv_smooth = (1 - smooth) * inv_interp + smooth * inv
    out = torch.where(wavelen > low_wl, inv_interp, inv)                       # long λ: interpolate
    return torch.where((wavelen >= high_wl) & (wavelen <= low_wl), inv_smooth, out)
```

**⚠ Trap:** loading this checkpoint in a runtime that does not implement `rope_type: "llama3"` and silently falling back to plain RoPE with base 500,000. Older inference stacks did exactly this. You get no warning, short context is perfect, and long context is bad in the specific "knee at 8k" way. **The check I run on any new serving stack: generate the `inv_freq` tensor from the framework and diff it against the reference computation above.** If the last 28 entries are not 8× smaller, the scaling is not being applied.

### Someone bumped `rope_theta` in a deployed model config and now our prefix cache hit rate is unchanged but quality dropped. Walk me through the debug.

Unchanged hit rate with dropped quality is the specific signature I want you to recognize: **the cache is being hit and the hits are wrong.** If quality had dropped because the cache was invalidated, hit rate would have fallen. It did not, so the cache is serving KV blocks computed under the old rotary geometry to a model now running the new geometry, and the two are silently incompatible.

The mechanism: prefix cache keys are hashes over token IDs (plus block position, plus usually LoRA ID and a few flags). `rope_theta` is a *model* property, not a *request* property, so it does not appear in the block hash. Within a single process this is harmless — the process either has the old base or the new one, and its own cache is self-consistent. It becomes a correctness bug the moment cache state outlives the config, and there are three concrete ways that happens:

1. **Rolling deploy with a shared or offloaded cache.** If you run CPU-offload KV, a distributed KV store (LMCache-style), or disaggregated prefill where prefill workers and decode workers are separate deployments, a rolling update leaves old-base blocks in a store being consumed by new-base workers. Every hit is corrupt.
2. **Disaggregated prefill mid-rollout.** Prefill pods updated, decode pods not (or vice versa). The KV crossing the wire was rotated with one base and is attended with the other. This produces *exactly* "hit rate normal, quality bad."
3. **A warm cache surviving a hot config reload** in any engine that supports reloading model config without dropping cache state.

The debug procedure: (a) confirm both fleets report the same `rope_theta` and `rope_scaling` — check the running process, not the repo, because the config may be baked into an image; (b) drain and cold-start one replica, route a canary at it, and compare per-position NLL against the warm fleet — if the cold replica is fine, it is a cache-poisoning problem, not a config problem; (c) look at whether quality degrades *with prefix length*, which it will, since longer shared prefixes mean more corrupt blocks.

**The fix, and this is the reviewable rule: the prefix-cache key must include a model fingerprint.** Hash the model weights identity plus the full rotary configuration (`rope_theta`, `rope_scaling` dict, `d_head`) into a namespace prefix on every cache key. Then a base change *invalidates* the cache instead of corrupting it, and you take a cold-start cost — which is a latency and dollar hit you can measure and plan for — instead of a silent quality regression you find out about from users.

**💰 Math:** the cold start you are choosing instead. With a 12k-token shared prefix, 200k calls/day, and a 90% cache-hit rate steady state, a full invalidation costs one uncached prefill for each distinct prefix until the cache refills. If you have 500 distinct system prompts and the cache refills within the first minutes, the one-time cost is 500 × 12,000 = 6M tokens × $3/1e6 = **$18**. Eighteen dollars to avoid a silent quality incident. Anyone who argues against namespacing the cache key on cost grounds has not done this arithmetic.

**🗣 Say this in the room:** "Hit rate flat plus quality down means the cache is hitting and the hits are stale. `rope_theta` isn't in the block hash because it's a model property, so any cache state that outlives the config — CPU offload, a shared KV store, disaggregated prefill mid-rollout — serves keys rotated under the old geometry. I'd namespace cache keys with a model+rope fingerprint so a base change invalidates instead of corrupting."

### Design the training stage that takes an 8k model to 128k. Data mix, curriculum, replay — I want the whole recipe.

Frame it as continued pretraining with a length curriculum, not as fine-tuning, because that framing gets the compute budget right. The model does not need to learn new knowledge; it needs to learn to *use* positional geometry it has never exercised, and to not forget everything else while doing it.

**Stage the length, don't jump.** Meta's Llama 3 report describes increasing context in six stages from 8k to 128k rather than one jump, and that is the right shape. Something like 8k → 16k → 32k → 64k → 128k, advancing only when two conditions hold: short-context evals have recovered to baseline, and a needle-style retrieval probe is saturated at the current length. Jumping straight to 128k means every gradient step early in training is computed on a model that is badly out of distribution, and you burn budget fighting instability instead of learning.

**The data mix is where this is won or lost.** The naive mix is "all long documents," which is wrong twice over. First, genuinely long natural documents are scarce — books, legal filings, long code repos, scientific papers — and if you train only on them you shift the domain distribution hard and the model gets worse at everything else. Second, most long documents have weak long-range dependencies: page 300 of a novel rarely *requires* page 4. Training on them teaches the model that distant context is safely ignorable, which is precisely the opposite lesson. My mix:

- ~30–40% naturally long documents (books, full repos with cross-file references, long-form technical docs). Code repositories are disproportionately valuable because cross-file symbol references are *real* long-range dependencies with a verifiable ground truth.
- ~20–30% **synthetically constructed long-dependency data**: multi-document QA where the answer requires facts placed at controlled distant positions; long-context summarization with citations; "here are 40 documents, answer using documents 7 and 31"; retrieval-augmented traces where the supporting evidence is deliberately buried mid-context. This is what actually teaches long-range attention.
- ~10% **document packing** of short documents to fill the window, with correct intra-document attention masking via `cu_seqlens` so tokens cannot attend across the boundary. This keeps the model exposed to full-length sequences cheaply.
- ~30% **replay of the original short-context pretraining mix**. This is the anti-forgetting term and it is not optional.

**Replay is the thing juniors skip.** Without it, short-context benchmarks regress several points and nobody notices until after release, because the whole eval effort was pointed at the long-context claim. My acceptance gate is: short-context evals within noise of the base model, measured on the *same* harness with the *same* seeds, reported in the same table as the long-context wins.

**Apply the rotary scaling before the extension stage, not after.** Train *with* the new `rope_scaling` in place so the model adapts to the compressed angles. Applying PI/YaRN to a checkpoint that was continued-pretrained under the old geometry gets you the worst of both.

**⚠ Trap:** using packed short documents as the majority of your long data because it is cheap and it fills the window. The sequence length is 128k and the model learns nothing about 128k dependencies, because the attention mask forbids crossing document boundaries. Your loss curve looks great, your GPU utilization looks great, and your retrieval eval is flat. If someone shows me a long-context extension run where the mix is mostly packed short docs, my first question is what fraction of tokens have a genuine dependency more than 32k tokens away.

### What does that extension stage cost? Show me the arithmetic.

The thing to get right is that long-context training is *superlinearly* more expensive per token, and the reason is that the attention term stops being a rounding error.

Standard training FLOP accounting is `C ≈ 6ND` for `N` parameters and `D` tokens — that covers the parameter matmuls (2N forward, 4N backward). The attention score computation is *not* in that term, because it does not involve parameters. Per token, causal attention costs roughly `2 · n_layers · d_model · T` FLOPs forward (the `QK^T` and `AV` matmuls, halved for causality), and about 3× that for forward+backward, so ≈ `6 · n_layers · d_model · T`.

Run it for Llama-3-8B: `n_layers = 32`, `d_model = 4096`, `N = 8e9`.

- Parameter term: `6N = 6 × 8e9 = 4.8e10` FLOP/token. Constant in `T`.
- Attention term at `T = 8,192`: `6 × 32 × 4096 × 8192 = 6.44e9` FLOP/token. That is 13% of the parameter term — a rounding error, which is why nobody thinks about it at short context.
- Attention term at `T = 131,072`: `6 × 32 × 4096 × 131072 = 1.03e11` FLOP/token. That is **2.15× the parameter term**.

Total per token: 5.44e10 at 8k versus 1.51e11 at 128k. **Long-context tokens cost 2.8× as much as short-context tokens for this model.** For a 70B (80 layers, `d_model` 8192) the parameter term grows to 4.2e11 and the attention term at 128k to 5.15e11 — a ratio of ~2.2×, so the effect is comparable but slightly milder for larger models, which is a genuinely useful thing to know: long context is relatively cheaper on bigger models.

**💰 Math — dollars.** Say 100B tokens of extension training at 128k on the 8B. Total FLOPs = 1.51e11 × 1e11 = **1.51e22**. An H100 does ~990 TFLOP/s dense bf16; assume 40% MFU (long-context runs have lower MFU because of sequence-parallel communication), so 3.96e14 effective FLOP/s. Time = 1.51e22 / 3.96e14 = 3.81e7 GPU-seconds = **10,600 GPU-hours**. At $2.50/GPU-hour (**📅 Volatile — see §5**) that is **~$26,500**, or about 55 hours of wall clock on a 192-GPU cluster. Doing the same 100B tokens at 8k would have cost 5.44e10 × 1e11 / 3.96e14 = 1.37e7 s = 3,800 GPU-hours ≈ $9,500.

The conclusion that matters: **$26k is not the reason teams skip long-context extension.** The compute is affordable. What is expensive is the data curation, the eval infrastructure to prove it worked, and the risk of a short-context regression you did not measure. When a candidate says "we couldn't afford to extend context," I push back with this arithmetic.

### How do you know the extension worked? Design the acceptance suite.

Four tiers, and I would not ship without all four. The framing I use: needle tests prove the *plumbing* works; they do not prove the *capability* exists.

**Tier 1 — per-position NLL curve.** Take held-out real documents of at least the target length, run a forward pass, and plot mean negative log-likelihood as a function of token position, bucketed every 4k. What you want is a curve that decreases monotonically (more context should always help predict the next token) and is flat-to-slightly-declining out to the full length. What a failed extension looks like: a knee at the original training length followed by a climb. This is the cheapest, fastest, most diagnostic signal you have and it runs in minutes. It is necessary and nowhere near sufficient — a model can have a beautiful NLL curve and be unable to retrieve anything, because NLL is dominated by local token statistics.

**Tier 2 — synthetic retrieval, but the hard variants.** Single-needle NIAH is a smoke test; treat a non-perfect score as a build failure and a perfect score as meaning nothing. What you actually run is the RULER-style family: multi-needle (place 4–8 needles, require all of them), multi-hop (needle A names the location of needle B), needle-in-a-needlestack (distractor needles with the same surface form), variable tracking (chase a chain of assignments), and aggregation (count or sum over the whole context). RULER's contribution was exactly this — showing that models saturating single-needle NIAH at their advertised length fall apart on multi-needle and multi-hop far earlier.

**Tier 3 — real-task long-context evals on your own corpus.** Public suites (LongBench, ∞Bench, HELMET-style batteries) are useful as a sanity check against other models, but the decision-relevant number is on *your* documents. Build 200–500 examples from real customer artifacts where the answer provably requires information at a controlled distance, and grade with exact-match or a rubric judge with a measured judge-agreement number.

**Tier 4 — short-context non-regression.** The same eval harness, same seeds, on your standard short benchmarks and your production traffic replay. This is the gate that catches the PI resolution loss.

**🗣 Say this in the room:** "Per-position NLL out to the full length as the smoke test, RULER-style multi-needle and multi-hop for capability, a domain suite on our own documents for the decision, and a short-context non-regression table so the extension doesn't quietly cost us the 95% of traffic that's under 2k. Single-needle NIAH I treat as a build check, not an eval."

### Where does long-context training data actually come from? Long documents are rare.

This is the real bottleneck and it is worth being blunt about it. The internet is mostly short. Common Crawl documents are overwhelmingly under 2k tokens. If you filter your pretraining corpus to documents over 64k tokens you are left with a small, weird, heavily-skewed slice: books (mostly old, mostly public domain, mostly narrative), legal and regulatory filings, scientific papers with long appendices, and code repositories. Training on that slice alone shifts your domain distribution violently.

Worse, natural long documents have *weak* long-range dependencies. Predicting token 100,000 of a novel almost never requires token 500. The mutual information between distant positions in natural text decays fast. So even if you had unlimited long documents, they under-teach the exact capability you are extending for. This is the single most important thing to say about long-context data and most candidates do not know it.

So the recipe is construction, not collection:

- **Repository-level packing with dependency ordering.** Concatenate the files of a real repo in an order that maximizes cross-file references (definitions before uses, or the reverse to force back-references), as one document. The long-range dependencies are real — a symbol used on line 90,000 was defined on line 300 — and verifiable.
- **Synthetic multi-document QA.** Take a retrieval corpus, sample `k` documents, plant an answer in one of them at a controlled position, and generate a question that requires it. Vary `k`, vary the position, include multi-hop chains that force two distant lookups.
- **Long-form summarization and citation tasks** where the target requires spanning the whole input.
- **Reordering and permutation objectives** — shuffle sections and ask the model to restore order, which cannot be done locally.
- **Concatenating related short documents** — same topic, same entities — so that cross-document dependencies exist even though each piece is short. This is strictly better than packing unrelated documents, and it does not need cross-document masking because you *want* attention to cross.

**⚠ Trap:** the synthetic data pipeline is where positional bias gets baked in. If you generate needles by sampling a uniform random position, you get roughly uniform coverage — good. If you generate by "insert the answer into a random one of the `k` documents" and your documents have a length distribution, you have oversampled positions inside long documents. And if a code generator always places the planted fact near a section boundary, the model learns to attend to section boundaries, which passes your eval and fails on real prose. **Always plot the position histogram of your planted facts before you train on them**, and always evaluate at positions your training data did not favor.

### Explain sliding-window attention. What does it buy you and what does it cost?

Sliding-window attention restricts each query to attend only to the previous `W` keys instead of all previous keys, so the mask becomes a band rather than a triangle. Mistral 7B shipped it with `W = 4096`. The mental model is a bounded-buffer stream processor: you stop keeping unbounded per-connection state and accept that anything older than `W` is only reachable indirectly.

"Indirectly" is doing real work in that sentence, and it is the part interviewers probe. A single layer sees `W` tokens back. But layer 2's tokens have each already aggregated `W` tokens of history, so layer 2's receptive field is `2W`. Stack `L` layers and the theoretical receptive field is `L × W` — for Mistral 7B, 32 × 4096 = 131,072 tokens. This is the same argument as stacked dilated convolutions in a CNN and it is why sliding-window models are not simply blind past `W`.

But theoretical receptive field is not effective receptive field. Information from 100k tokens back has to survive 25 hops of lossy aggregation through the residual stream, competing with everything else written there. It works fine for *diffuse* signals (topic, style, general subject matter) and badly for *precise* recall (an exact account number mentioned once, 100k tokens ago). That asymmetry — good at gist, bad at exact retrieval — is the defining behavioral signature of window-limited attention, and it will show up in your evals as decent summarization scores and terrible needle scores.

**📐 Numbers you must know — the cache win.** KV bytes per token for Mistral-7B (32 layers, 8 KV heads, `d_head` 128, bf16) is `2 × 32 × 8 × 128 × 2 = 131,072 B = 128 KiB/token`. Under full attention at 32k context, one sequence needs 32,768 × 128 KiB = **4 GiB**. Under a 4,096 window, the cache is *capped* at 4,096 × 128 KiB = **512 MiB — regardless of how long the conversation gets.** That is an 8× reduction at 32k and a 32× reduction at 128k, and crucially it converts KV memory from `O(T)` to `O(1)`, which is what makes unbounded-length streaming feasible at all. It also converts attention compute per token from `O(T)` to `O(W)`.

**⚠ Trap:** assuming your serving engine actually implements the rolling-buffer eviction. Several engines historically allocated the full context worth of KV and merely applied the band mask, giving you the compute win and none of the memory win. Check the engine's memory profile at 64k, not its docs. If cache memory grows linearly past `W`, you are paying for full attention and getting sliding-window quality — the worst possible trade.

### Gemma uses interleaved local and global attention layers. Work out the cache savings and tell me why the ratio matters.

The design: instead of every layer being sliding-window (which caps recall) or every layer being global (which caps memory), interleave them. Gemma 2 alternated local and global layers roughly 1:1 with a 4,096-token window; Gemma 3 moved to a much more aggressive ratio — on the order of 5 local layers per 1 global layer with a smaller window — precisely to attack KV memory at long context. (**📅 Volatile — see §5:** exact ratios and window sizes vary by model generation; read them out of the config, do not recite.)

The mental model that makes this feel inevitable: **you do not need every layer to be able to reach across the document; you need enough of them.** Global layers are the long-range highway; local layers do the dense local work. Since KV cache cost is summed over layers, converting a layer from global to local at `W = 1024` reduces that layer's contribution from `T` tokens to `min(T, 1024)`.

Work the arithmetic at `T = 131,072` for a hypothetical model with `L` layers. Under all-global, cache is proportional to `L × 131,072` token-slots. Under a 5:1 local:global split with `W = 1024`:

```
effective slots per layer = (1/6)·131,072 + (5/6)·1,024
                          = 21,845 + 853
                          = 22,698
```

versus 131,072 for all-global — a **5.8× reduction in KV memory**, which translates almost one-for-one into concurrent sequences per GPU. At a 1:1 ratio with `W = 4096` you would get `(0.5 × 131,072) + (0.5 × 4,096) = 67,584`, only a 1.94× reduction — which is why the ratio moved.

Why the ratio matters beyond arithmetic: each global layer is a full-bandwidth long-range channel, and there is a floor below which multi-hop long-range reasoning breaks, because a two-hop retrieval needs at least two global layers *after* the relevant information has been assembled. Too few global layers and you get a model that can find a fact but cannot find a fact whose location depends on another fact. My rule of thumb when evaluating one of these models: run multi-hop NIAH specifically, because that is the eval the ratio choice shows up in first.

**💰 Math:** at 128 KiB/token and 131,072 tokens, one all-global session needs **16 GiB** of KV. An 80 GB H100 with ~40 GiB free after weights and activations therefore holds 40/16 = **2.5 concurrent 128k sessions**. The 5:1 interleave cuts per-session KV to 16/5.8 = **2.76 GiB**, giving 40/2.76 ≈ **14.5 sessions**. Serving 1,000 concurrent long sessions goes from 1,000/2.5 = **400 GPUs** to 1,000/14.5 = **69 GPUs**. At $2.50/GPU-hour that is 331 GPUs × 730 h × $2.50 = **~$604k/month saved** by an architecture choice made before a single token was trained.

### I give you a closed-weight API model and no training budget. How far can you push its context, and what do you do instead?

Zero. You cannot apply YaRN to an API. This question is a judgment test — the interviewer wants to see whether you reach for the training lever when you have no training access, and whether you know the inference-side toolkit.

What you actually have is four levers, in the order I would apply them.

**Lever one: find the usable context, don't trust the advertised one.** Run a multi-needle and multi-hop probe on your own documents at 8k, 16k, 32k, 64k, and the advertised max. You will typically find accuracy holding until somewhere well below the advertised limit and then falling off. Set your production context budget at the last length where you were above your accuracy bar, not at the API's limit. Quote both numbers in every design doc: "advertised 200k, usable 45k on our contract corpus at 92% multi-needle."

**Lever two: reduce what you put in, not increase what it can hold.** This is retrieval, and it is almost always the right answer on cost grounds alone. Retrieving 8k of relevant context instead of stuffing 150k costs 18.75× less per call and cuts TTFT by roughly the same factor, because prefill is linear-plus-quadratic in input length.

**Lever three: restructure the context for the model's positional biases.** Given lost-in-the-middle, put the most important retrieved evidence at the *start* and at the *end* of the context, not the middle. Put the question both before and after the documents. These are free and measurably worth several points on long-context QA.

**Lever four: decompose.** Map-reduce over chunks with a synthesis pass, or an agentic loop that reads a table of contents and then fetches sections on demand. This trades latency and token count for accuracy and is the right call when the task genuinely requires the whole corpus.

**🗣 Say this in the room:** "With an API I have no positional levers at all — YaRN and PI need weight access. What I do is measure the *usable* context on my own corpus rather than trusting the advertised number, then spend my effort on retrieval and context ordering, because putting 8k of the right tokens in front of the model beats 150k of everything at both cost and accuracy."
