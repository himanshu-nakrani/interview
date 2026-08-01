### Start me at the beginning — why do language models tokenize at all? Why not just feed them raw bytes or raw characters?

Tokenization is a compression codec sitting between your string and the model's embedding table, and it exists because the transformer's cost is quadratic in sequence length and linear in vocabulary size — so you are trading one axis against the other. Every token costs you `O(T²)` attention work and `O(T·d)` of KV cache; every vocabulary entry costs you one row in the embedding matrix and one row in the unembedding matrix, plus a slice of softmax time. Tokenization picks the operating point on that curve.

Concretely: English text is roughly 4 characters per token under a modern 100k–200k BPE vocabulary. If you fed raw UTF-8 bytes instead, the same document would be ~4× longer. At 4× the sequence length you pay 16× the attention-score FLOPs, 4× the KV cache bytes, and 4× the wall-clock decode steps to emit the same text. That is not a rounding error — it is the difference between a 32k-token document fitting in context and a 128k-token document not fitting.

The other direction fails too. A word-level vocabulary needs hundreds of thousands of entries and still hits out-of-vocabulary words constantly — every typo, every proper noun, every new product name becomes `<UNK>` and the model literally cannot represent it. Subword tokenization is the compromise: frequent words get one token, rare words decompose into pieces, and nothing is unrepresentable.

**🗣 Say this in the room:** "Tokenization is lossy-free compression that buys you sequence length at the cost of vocabulary size. Attention is quadratic in tokens and the embedding table is linear in vocab, so you're picking a point on that trade curve. Bytes are maximally general and maximally expensive; words are compact and can't handle the tail. Subword BPE is the settled compromise."

The backend analogue that actually helps: the tokenizer is a *dictionary-coder*, like the LZ78 family. It builds a table of frequently-recurring byte sequences during training and then encodes new input against that fixed table. The important consequence — and the source of half this section's failure modes — is that **the table is frozen at training time and the model's entire world is expressed in it.** If your production text distribution drifts away from the tokenizer's training distribution, you do not get an error; you get silently worse compression, more tokens, higher bills, and degraded quality on the tokens that fall apart.

### Implement BPE training from scratch. I want the merge-learning loop, and I want to see it run.

The mental model first: BPE training is a greedy hill-climb on compression. You start with every byte as its own symbol, count every adjacent symbol pair in the corpus, merge the single most frequent pair into a new symbol, and repeat. Each merge shortens the corpus by exactly the count of that pair. After `k` merges you have a vocabulary of `256 + k` symbols and an ordered merge list — and the order is the whole algorithm, because encoding replays it.

**📄 Paper:** Sennrich, Haddow & Birch (2016), "Neural Machine Translation of Rare Words with Subword Units" — took Gage's 1994 byte-pair *compression* algorithm and repurposed it as an NMT vocabulary builder, replacing word-level vocabularies with their permanent UNK problem.

```python
from collections import Counter

def train_bpe(word_freqs: dict[str, int], num_merges: int):
    """word_freqs: pre-tokenized word -> corpus frequency."""
    # Every word starts as a tuple of single bytes.
    words = {tuple(bytes([b]) for b in w.encode("utf-8")): f
             for w, f in word_freqs.items()}
    merges = []
    for _ in range(num_merges):
        pairs = Counter()
        for syms, f in words.items():
            for a, b in zip(syms, syms[1:]):
                pairs[(a, b)] += f
        if not pairs:
            break
        # max on (count, pair) so ties break deterministically — see the trap
        best = max(pairs, key=lambda p: (pairs[p], p))
        merges.append(best)
        words = {_apply(syms, best): f for syms, f in words.items()}
    return merges          # ordered: index == merge rank

def _apply(syms, pair):
    a, b = pair
    out, i = [], 0
    while i < len(syms):
        if i + 1 < len(syms) and syms[i] == a and syms[i + 1] == b:
            out.append(a + b); i += 2
        else:
            out.append(syms[i]); i += 1
    return tuple(out)
```

Two structural points worth saying out loud. First, **merges never cross word boundaries** — that is why the input is a dict of pre-tokenized words rather than a single string, and it is why the pre-tokenization regex (below) is load-bearing rather than cosmetic. Without it, BPE would happily learn a single token for `" the quick brown"` and your vocabulary would be full of phrase fragments that generalize terribly.

Second, **the merge list is ordered and the order is the model.** `merges[0]` is applied before `merges[1]` always, everywhere, forever. Ship a merge list in a different order and you have a different tokenizer, even with an identical vocabulary set.

**⚠ Trap:** non-deterministic tie-breaking. `max(pairs, key=pairs.get)` returns whichever equally-frequent pair Python's dict happened to yield first, which depends on insertion order, which depends on corpus shard ordering. Train the "same" tokenizer twice on shuffled shards and you get two different merge lists that produce different token IDs for the same text. If you have already trained a model against list A and rebuild list B for serving, every embedding row is now wrong and quality collapses in a way that looks like a checkpoint corruption bug. Sort the tie-break key explicitly, and pin the tokenizer artifact in the model repo, not in a build step.

**🏋 Drill:** 25 minutes, no autocomplete, no reference. Write `train_bpe` and `_apply` from memory, run 50 merges over the text of this paragraph, and print the ten highest-rank merges. Pass criterion: the merges are legible English fragments (`th`, `in`, `er`, `the`), your byte-level round-trip is exact, and you did not need to look anything up.

### Given a trained merge list, how does encoding actually work? Walk me through it, because most people get this wrong.

Here is the thing people get wrong: encoding is **not** "scan left to right and merge whatever you can." Encoding replays the training merges *in rank order*, globally, over the whole word. At each step you find the lowest-rank (earliest-learned) applicable merge anywhere in the current symbol sequence and apply it. Then you look again. You stop when no learned merge applies.

```python
INF = float("inf")

def encode_word(word: str, ranks: dict[tuple[bytes, bytes], int]) -> list[bytes]:
    syms = [bytes([b]) for b in word.encode("utf-8")]
    while len(syms) > 1:
        rank, i = min(
            ((ranks.get((syms[j], syms[j + 1]), INF), j) for j in range(len(syms) - 1)),
            default=(INF, 0),
        )
        if rank is INF or rank == INF:
            break
        syms[i:i + 2] = [syms[i] + syms[i + 1]]
    return syms
```

That is the entire encoder. `ranks` is `{pair: index}` built from the merge list. The full pipeline is: pre-tokenization regex splits the input into word-ish chunks → each chunk is encoded independently by the loop above → each resulting byte string is looked up in the vocab to get an integer ID.

Two consequences that matter in interviews. **BPE encoding is greedy, not optimal.** It does not find the minimum-token segmentation of a string; it finds the segmentation implied by replaying the training merges. There exist strings where a different segmentation would use fewer tokens and BPE will not find it. This is why the same word can tokenize differently depending on what surrounds it and whether a leading space is present.

**Encoding is deterministic given the merge list.** Same input, same merges, same IDs, every time, in every implementation that implements this correctly. That determinism is what lets you cache prompt prefixes by token ID and what makes prefix-cache hit rates reproducible.

**⚠ Trap:** the O(n²) naive encoder. The loop above rescans every pair on every merge, so a 2,000-character pathological input (a base64 blob, a minified JS bundle, a long hex string) can take quadratic time in the pre-token length. Production tokenizers cap pre-token length precisely to bound this. If your ingest service hangs on one document out of a million, look at the tokenizer before you look at the model — I have seen a document-processing pipeline stall for 40 seconds on a single 200KB unbroken base64 attachment because the pre-tokenizer had no length cap and the Python-side BPE loop went quadratic.

### What is byte-level BPE, and why does everyone say it "removes the UNK token"?

Byte-level BPE means the alphabet — the leaf symbols before any merges — is the 256 possible byte values, not the set of Unicode characters seen in the training corpus. Every possible input string is a sequence of bytes, and every byte is in the vocabulary, therefore **every possible string is representable**. There is no input for which the tokenizer must emit `<UNK>`. Emoji you have never seen, Klingon, corrupted binary, a novel Unicode plane added in 2027 — all of it encodes, possibly inefficiently, but losslessly.

**📄 Paper:** Radford et al. (2019), GPT-2 — introduced byte-level BPE for LM pretraining, replacing character-level vocabularies that needed an UNK escape hatch and a separate handling path for rare scripts.

Contrast with character-level BPE (which SentencePiece can also do): its alphabet is the characters observed during training, plus an explicit `<unk>` for anything else. That means a Unicode character absent from the training corpus is *unrepresentable*. In a multilingual product this is not hypothetical — it is a Tuesday.

The one wrinkle worth knowing: GPT-2's implementation applies a reversible byte→printable-Unicode mapping so the vocabulary file can be stored as readable text. That is why you see `Ġ` for space and `Ċ` for newline in GPT-2/GPT-4-family vocab dumps. `Ġ` is not a real character in your text; it is byte `0x20` displayed through the mapping. Modern `tokenizers`-library artifacts still show it.

**⚠ Trap:** believing "no UNK" means "no degradation." Byte fallback guarantees *representability*, not *quality*. A Devanagari string absent from the pretraining mix still encodes — as one token per byte, at 3 bytes per character. It is representable, it is 3× more expensive, and the model has seen those byte sequences in that combination essentially never, so its behaviour on them is undefined in the way an untrained embedding is undefined. "It didn't error" and "it works" are different claims. The failure is a silent quality-and-cost failure, not a crash.

**💰 Math:** a 500-character Hindi support message. Under a tokenizer with no Devanagari merges, worst case is one token per UTF-8 byte: Devanagari is 3 bytes/char, so 1,500 tokens. The same content in English is ~125 tokens. At $3/Mtok input that is $0.0045 vs $0.000375 per message — 12×. Across 2M messages/month: **$9,000 vs $750**. Byte fallback saved you from a crash and cost you $8,250 a month.

### SentencePiece Unigram versus BPE — what's actually different, and when would you pick one?

Different search direction, same goal. **BPE is bottom-up and greedy**: start from bytes, repeatedly merge, keep the merge list. **Unigram is top-down and probabilistic**: start from a large seed vocabulary of candidate substrings, fit a unigram language model over pieces via EM, then iteratively *prune* the pieces whose removal costs the least likelihood, until you hit the target vocab size.

**📄 Paper:** Kudo (2018), "Subword Regularization: Improving Neural Network Translation Models with Multiple Subword Candidates" — introduced the unigram LM subword model and the idea of sampling among segmentations during training. Kudo & Richardson (2018) is the SentencePiece library that ships it.

The mechanical consequences:

*Unigram keeps probabilities, so it can score segmentations.* At encode time it runs Viterbi to find the maximum-likelihood segmentation of the string given piece log-probs. That is a genuine optimum under its model — unlike BPE, which just replays merges. It also means Unigram can *sample* alternative segmentations proportional to their probability, which is **subword regularization**: train with a different segmentation of the same word each epoch and the model becomes robust to segmentation noise. BPE's analogue is BPE-dropout (Provilkov et al., 2020), which randomly skips merges during training.

*SentencePiece treats the input as a raw stream, not as pre-split words.* It escapes spaces as `▁` (U+2581) and includes them in pieces. This means it is genuinely language-agnostic — it works on Japanese and Thai, which have no whitespace word boundaries, without a language-specific pre-tokenizer. That is the actual reason SentencePiece became the multilingual default.

**When I pick what:** BPE for English-and-code-heavy monolingual-ish models where I want maximum compression and a simple deterministic encoder — and it is what the GPT and Llama-3 lineages use. Unigram (via SentencePiece) when the corpus spans scripts without whitespace segmentation, when I want subword regularization for a smaller model or a low-resource language, or when I want the tokenizer to be robust to how the text was normalized. Honestly, at frontier scale the quality delta between them is small and inconsistently reported; the *engineering* difference — whitespace handling, byte fallback config, reversibility — matters more than the algorithm.

**⚠ Trap:** assuming SentencePiece implies Unigram. SentencePiece is a *library* with both `--model_type=bpe` and `--model_type=unigram`. Llama 1/2 used SentencePiece in **BPE** mode. Saying "Llama 2 uses SentencePiece so it's Unigram" is a confident wrong answer, and it's a common one.

### How does WordPiece choose its merges, and why is it different from BPE?

BPE merges the pair that occurs most *often*. WordPiece merges the pair that most increases the *likelihood of the corpus* under a unigram language model over pieces. In practice that reduces to picking the pair maximizing

```
score(a, b) = freq(a, b) / (freq(a) · freq(b))
```

which is a pointwise-mutual-information-flavoured criterion rather than a raw count. The intuition: `freq(a,b)` alone rewards merging pieces that are individually ubiquitous — merging `e` and `s` looks great because both are everywhere. Dividing by the marginals asks the sharper question, "do these two pieces co-occur more than chance would predict?" That biases WordPiece toward merging pieces that are *specifically* bound to each other, producing more morpheme-like units.

**📄 Paper:** Schuster & Nakajima (2012), "Japanese and Korean Voice Search" — the original WordPiece formulation; it reached the mainstream via BERT and Google NMT.

The other WordPiece signature is its inverted continuation marking: instead of marking word-initial pieces with a leading space, it marks word-*internal* pieces with `##`. `tokenization` → `token`, `##ization`. And its encoder is greedy longest-match-first from the left over the vocabulary, not a merge replay — a different algorithm from BPE's, though it usually lands somewhere similar.

Practically: WordPiece lives in the BERT/DistilBERT/ELECTRA encoder lineage. If you are fine-tuning a BERT-family reranker or a classical NER model you will meet it. Essentially no modern decoder-only generative model uses it. Know the merge criterion — it is a clean two-minute answer that separates "read the docs" from "read the paper" — and know that `##` is the tell when you are staring at an unfamiliar tokenizer's vocab file.

### Walk me through GPT-4's pre-tokenization regex. What is each clause doing, and why does pre-tokenization exist at all?

Pre-tokenization is the step *before* BPE that splits raw text into chunks; BPE then runs independently inside each chunk and merges never cross a chunk boundary. It exists to impose linguistic structure that pure frequency counting would otherwise destroy. Without it, BPE would learn tokens spanning multiple words and punctuation, the vocabulary would fill with phrase-shaped junk, and the model would generalize badly to any phrasing it had not memorized.

The `cl100k_base` pattern (GPT-4 / GPT-3.5-turbo era) is, clause by clause:

```
(?i:'s|'t|'re|'ve|'m|'ll|'d)      # English contraction suffixes, case-insensitive
|[^\r\n\p{L}\p{N}]?\p{L}+          # optional leading non-letter/digit, then a letter run
|\p{N}{1,3}                        # digits, in chunks of AT MOST 3
| ?[^\s\p{L}\p{N}]+[\r\n]*         # optional space, then punctuation/symbols, trailing newlines
|\s*[\r\n]+                        # newline runs
|\s+(?!\S)                         # trailing whitespace not followed by non-space
|\s+                               # any remaining whitespace
```

The design decisions worth articulating:

**The leading-space clause** (`[^\r\n\p{L}\p{N}]?\p{L}+` and `\s+(?!\S)`) is why `" hello"` and `"hello"` are *different tokens* in this family. Attaching the space to the following word is what makes English so compact — most words in running text are preceded by a space, so one token covers both. It is also the root of the trailing-whitespace bug that bites people in production.

**The `\p{N}{1,3}` clause** hard-caps numeric runs at three digits. This is a deliberate change from GPT-2, whose pattern was `\p{N}+` — unbounded — and therefore learned arbitrary whole-number tokens like `2017` and `1000000` based on how often they appeared in the corpus. Capping at 3 makes the numeric vocabulary bounded and regular.

**Whitespace clauses are ordered so that a run of spaces followed by text splits differently from trailing spaces.** `\s+(?!\S)` catches trailing whitespace (end of string or before a newline) as its own token, so that `"foo   "` does not produce a `"   f"`-shaped fragment.

**📅 Volatile:** the exact pattern differs by tokenizer generation — `o200k_base` extends the contraction handling and Unicode classes, and Llama 3's tiktoken-style pattern is close to `cl100k_base` but not byte-identical. Read the pattern out of the artifact you are actually shipping (`tokenizer.json`'s pre-tokenizer section, or the tiktoken encoding definition) rather than reciting one. Verify before your loop.

**⚠ Trap:** treating pre-tokenization as an implementation detail you can swap. Changing the regex changes which merges are reachable, which changes every token ID downstream. A tokenizer is the tuple (normalizer, pre-tokenizer, merge list, vocab, post-processor) — all five, versioned together. I have seen a team port a tokenizer to a new runtime, faithfully copy the vocab and merges, and reimplement the pre-tokenizer "equivalently" with `\s+` handling subtly reordered. Result: about 0.3% of production strings tokenized differently, quality dropped a few points on long-context tasks, and it took two weeks to find because 99.7% of the eval set was unaffected.

### The digit grouping thing — why does chunking numbers into 1–3 digit pieces matter, and what does it do to arithmetic?

Start from the mechanism. Under `\p{N}{1,3}`, the number `1234567` pre-splits into `123`, `456`, `7` — grouped **left to right**, so the final group is the *least* significant digits and it is the one with a ragged size. Compare that to how a human does addition: right to left, aligning ones with ones. The tokenizer's grouping is misaligned with the algorithm the model has to learn.

Concretely, `1234 + 5678`. Left-to-right 3-grouping gives `123|4` and `567|8`. The model must learn that the `4` in `123|4` occupies the same positional column as the `8` in `567|8`, and that `123` and `567` are the thousands-and-up parts. Now add one digit to one operand — `12345 + 5678` becomes `123|45` and `567|8` — and the column alignment between the two operands changes completely. Every different digit-length combination is a *different* alignment problem. The model can and does learn this, but it learns it as memorization over the combinatorics rather than as a single carry rule, which is exactly why arithmetic accuracy falls off a cliff as digit count grows and why errors cluster at carries across chunk boundaries.

**📄 Paper:** Singh & Strouse (2024), "Tokenization counts: the impact of tokenization on arithmetic in frontier LLMs" — showed that the digit-chunking scheme materially changes arithmetic accuracy, with right-to-left grouping outperforming left-to-right grouping at equal model scale, and single-digit tokenization being a strong baseline. The contribution that matters for you: this is a *tokenizer* problem, not a reasoning problem.

The three schemes in the wild, and what each costs:

| Scheme | `1234567` → | Tokens for a 7-digit number | Arithmetic behaviour |
|---|---|---|---|
| Unbounded (`\p{N}+`, GPT-2 era) | one token, if seen in corpus | 1–7, unpredictable | Worst: frequency-dependent, wildly inconsistent |
| L2R 1–3 grouping (`cl100k`, Llama 3) | `123`,`456`,`7` | 3 | Good compression, misaligned columns |
| Single digit (some model families) | 7 tokens | 7 | Best arithmetic, 2.3× the tokens |

**🗣 Say this in the room:** "Digit grouping is a compression-versus-arithmetic trade. Three-digit left-to-right chunks cut numeric token count by about 3× versus per-digit, but they misalign place value across operands, so the model has to memorize alignment rather than learn a carry rule. If arithmetic accuracy actually matters to my product, I don't fight the tokenizer — I route the computation to a calculator tool and let the model write the expression."

That last sentence is the senior answer. The production fix for arithmetic is never "prompt harder"; it is a tool call. But there is a cheap intermediate: **comma-separate numbers in prompts.** `1,234,567` pre-splits on the commas into `1`, `,`, `234`, `,`, `567` — which forces 3-digit groups aligned to actual thousands boundaries, restoring column alignment across operands. It costs 2 extra tokens per number and measurably helps on multi-digit addition. Test it on your own eval before believing me.

### "How many r's in strawberry" — explain that failure mechanistically, not as a joke.

The model never sees the letters. `strawberry` under a typical BPE vocabulary is something like `str` + `aw` + `berry` — three integer IDs. Those IDs index into an embedding table, and each embedding is a dense vector learned from *distributional* co-occurrence: what words appear near `str`, what contexts `berry` shows up in. Nowhere in the forward pass is there a representation of "the token `berry` is composed of the characters b-e-r-r-y." The orthographic decomposition was discarded at the tokenizer boundary and is never recovered.

So when you ask for the count of `r`, you are asking the model to perform a character-level operation on an input it holds as three opaque symbols. It can only answer from whatever character-composition information leaked into the embeddings indirectly — from the pretraining corpus containing spelling discussions, acrostics, hyphenations, and typo corrections. That signal exists, which is why models often get short common words right, and it is weak and irregular, which is why they fail on the tail.

The precise analogy for a backend engineer: you have a table where the primary key is a hash of the string, and you have thrown away the string. You can answer "which rows are similar to this hash" beautifully, because you built an index on that. You cannot answer "how many `r`s are in the value" at all, because that requires the value and you only kept the hash. The model's "knowledge" that `berry` contains two `r`s is not stored anywhere structurally — it's a weak statistical residue.

Three predictions this mental model makes, which is how you demonstrate you actually understand it rather than reciting it:

1. **Character counting gets better when the word is rare enough to fragment heavily.** A word that tokenizes to five single-character pieces is far easier to count than a word that is one token.
2. **Spacing out the letters fixes it.** `s t r a w b e r r y` pre-tokenizes into individual letter tokens, and suddenly the model is doing a counting task over visible symbols. This works reliably and is a good demo.
3. **Reasoning models do better** — not because their tokenizer changed, but because chain-of-thought lets them *emit* the spelled-out form into their own context and then count over that. They are converting an impossible task into a possible one by writing the characters down.

**⚠ Trap:** citing this as evidence that "LLMs can't reason." It is evidence of an input-representation bottleneck at one specific layer of the stack, and it is fixable at that layer (character-level models, byte-level models) at a large compute cost that nobody has judged worth paying. The correct production response is to detect character-level tasks and route them to code — `len([c for c in w if c == 'r'])` is exact, free, and takes 3 milliseconds.

### How do I choose a vocabulary size? Show me what the trade-off actually costs.

Vocabulary size is the one hyperparameter that shows up in three separate budgets simultaneously: parameter count, sequence length, and softmax cost. Bigger vocab means better compression (fewer tokens per document, so shorter sequences, cheaper attention, more text per context window) and a bigger embedding/unembedding matrix (more parameters, more memory bandwidth per decode step, more softmax time).

The parameter side is arithmetic you should be able to do live. Embedding and unembedding are each `V × d_model`.

**📐 Numbers you must know:**
- Llama-3-8B: `V = 128,256`, `d = 4096` → `128256 × 4096 = 525.3M` parameters *per matrix*. Untied, so **1.05B of the 8.03B total is vocabulary — 13.1%.**
- Gemma-2-2B: `V = 256,000`, `d = 2304` → `589.8M`. Tied, so one copy, but that is **22.6% of a 2.61B model.**
- Qwen2-0.5B: `V ≈ 151,936`, `d = 896` → `136.1M`, tied → **27.6% of a 494M model.**

The pattern is the point: **vocabulary cost is a fixed absolute number, so its relative burden explodes as models shrink.** A 256k vocab on a 70B model is a rounding error. On a 500M model it is a quarter of your parameters, and those parameters are doing lookup, not computation. This is why small-model families sometimes ship a smaller vocab than their big siblings, and why "just use the biggest vocab" is wrong advice below ~3B parameters.

The compute side. At decode, the `lm_head` matmul is `2 · d · V` FLOPs. For Llama-3-8B: `2 × 4096 × 128256 = 1.05 GFLOP`, against a full forward pass of `2 × 8.03e9 = 16.1 GFLOP` — so **6.5% of decode FLOPs are spent on the output projection alone.** In memory-bandwidth terms it is the same fraction: 1.05GB of the 16.1GB of bf16 weights read per token. Add the softmax over 128k logits, plus top-k/top-p which needs a sort or partial sort over that vector, and the vocab tax on per-token latency is real but not dominant.

Now the benefit. Going from a 32k vocab to a 128k vocab typically cuts English tokens-per-document by roughly 10–15%, and cuts *code* and *non-English* token counts by considerably more — this is where the 32k→128k→150k+ industry drift came from. Fewer tokens is a multiplicative win: it reduces prefill FLOPs, decode steps, KV cache bytes, and your API bill, all at once.

**💰 Math — is the 32k→128k upgrade worth it on a 7B model?** The extra 96,000 rows × 4096 dims × 2 untied matrices = **786M extra parameters**, taking a 7.0B model to 7.8B — an 11% increase in weight memory and per-token bandwidth. Against that: if it cuts your token count 12%, you save 12% of every prefill, every decode step, and every KV cache byte. On a workload where sequences average 8k tokens, the sequence savings dominate the parameter cost by a wide margin. On a workload of 200-token classification requests where the KV cache is trivial and the model is bandwidth-bound on weights, the 11% weight increase is a straight 11% latency regression and the compression buys you almost nothing. **Short-sequence workloads want small vocabularies; long-context workloads want large ones.** That is the decision rule.

### Tied versus untied embeddings — what actually changes, and when does the choice matter?

Weight tying means the unembedding matrix is the transpose of the embedding matrix: one `V × d` tensor serving both "token ID → vector" on the way in and "vector → logits" on the way out. Untied means two independent matrices.

**📄 Paper:** Press & Wolf (2017), "Using the Output Embedding to Improve Language Models" (with Inan et al. 2016 concurrent) — showed that sharing input and output embeddings both halves the parameter count and improves perplexity in the small-model regime, by regularizing the input embeddings toward the output space.

The intuition for why tying is even coherent: both matrices are maps between token identity and the residual-stream space. The input embedding asks "what vector represents this token," the unembedding asks "how much does this residual vector look like this token." Under a dot-product readout those are geometrically the same question, so sharing the parameters is a reasonable prior, not a hack.

The decision rule I use:

**Tie when the vocabulary is a large fraction of your parameters** — small models, large vocabularies. Gemma-2-2B ties and saves 590M parameters, 22% of the model. Qwen's small models tie. At that scale the parameter savings are enormous and the regularization genuinely helps.

**Untie when the model is big enough that vocabulary is a small fraction** — Llama 3 unties at both 8B and 70B. At 8B, untying costs 525M extra parameters (6.5% of the model) and buys the model freedom to shape input and output token geometry independently. There is real evidence this helps at scale: the constraints on a good *input* representation (be a useful starting point for the residual stream) and a good *output* representation (be maximally separable under a dot product) are not identical, and forcing them to be identical costs you something once you can afford not to.

**The migration trap.** If you add domain tokens to a model and resize the embeddings, you must know which regime you are in. Under tying, resizing `embed_tokens` implicitly resizes `lm_head` — one edit, both sides. Under untying, they are separate tensors and you must resize both. Resize only the input embedding on an untied model and you get a shape mismatch at the logits layer, which at least errors loudly. The nastier version: some frameworks will *silently re-tie* on resize if the config's tie flag says so, so you resize an untied checkpoint, the framework helpfully ties it, and you have just replaced a trained `lm_head` with a transposed `embed_tokens`. Loss jumps, nobody knows why.

**⚠ Trap:** reading `tie_word_embeddings: true` in a config and assuming the tensors on disk are shared. They may be stored separately and tied at load. Verify with an identity check on the loaded tensors (`model.get_input_embeddings().weight.data_ptr() == model.get_output_embeddings().weight.data_ptr()`), not by reading the config. The rule I enforce in review: any code path that resizes a vocabulary asserts the tie state before and after, and asserts the two matrices' shapes match expectation.
