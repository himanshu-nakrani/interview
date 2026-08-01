### Draw me the timeline for static batching versus continuous batching, and tell me what the unit of scheduling is in each.

Static batching schedules at the granularity of a **request**: you gather 8 requests, run prefill on all 8, then decode in lockstep until *every* sequence has emitted its EOS, then return all 8 and admit the next 8. Continuous batching schedules at the granularity of an **iteration**: after every single forward pass, the scheduler re-evaluates the batch — finished sequences leave immediately and their slots are refilled from the waiting queue in the *same* step.

Draw it as two Gantt charts. In the static chart you see eight bars of wildly different lengths, and seven of them end early with dead grey space stretching to the right until the longest one finishes; the batch slot is occupied but idle. In the continuous chart the bars are ragged but the vertical column is always full — every row that ends is immediately replaced.

The backend analogue is exact and I would use it out loud: static batching is a thread pool that waits for **all** tasks in a submitted chunk to complete before pulling more work; continuous batching is a proper work-stealing pool with per-task completion. You already know which one you would ship.

The mechanical requirement that makes iteration-level scheduling possible is that decode is embarrassingly per-sequence: at step `t`, sequence A attending over 4,000 cached tokens and sequence B attending over 40 cached tokens are doing structurally identical work — one query vector each, gathered against their own KV. There is nothing that forces them to have the same length. Static batching only ever required it because the implementation padded to a rectangle.

**📄 Paper:** Yu et al. (2022), *Orca: A Distributed Serving System for Transformer-Based Generative Models*, OSDI. Introduced **iteration-level scheduling** plus **selective batching** — batching the linear layers across all sequences while running attention per-sequence, because attention is the only operator whose shape actually depends on sequence length. It replaced request-level static batching, and every serious engine since is a descendant.

**🗣 Say this in the room:** "The insight in Orca is that batching and scheduling do not have to happen at the same granularity. You batch the GEMMs, which want a big rectangle, and you schedule per iteration, which lets a finished sequence leave without waiting for the longest one in its cohort. Everything vLLM does on top of that is memory management."

### Make head-of-line blocking concrete for me. Batch of 8, one request generates 2,000 tokens, the rest generate 50.

Assume a 20 ms per-iteration step time for the whole batch (roughly right for a 70B on a 4-way H100 node at moderate concurrency).

**Static:** the batch runs for 2,000 iterations because it runs until the longest sequence finishes. Wall clock = 2,000 × 20 ms = **40 seconds**. The seven short requests finished their real work at iteration 50 — 1 second in — but their responses are held (or, if you stream, their *slots* are held) for 39 more seconds. Total useful tokens produced = 2,000 + 7 × 50 = 2,350. Total slot-iterations consumed = 8 × 2,000 = 16,000. **Utilization = 2,350/16,000 = 14.7%.** You paid for 16,000 sequence-steps of GPU memory bandwidth and got 2,350 tokens.

**Continuous:** at iteration 50 the seven short sequences complete and are replaced from the queue. Over the same 40-second window you run 2,000 iterations with a full batch of 8 the whole time, producing ~16,000 tokens instead of 2,350. That is a **6.8× throughput difference on identical hardware with an identical model**, arising entirely from a scheduling decision.

The latency story is separate and equally important. Under static batching, the p99 TTFT of a *short* request is dominated by the tail of the generation-length distribution of whatever cohort it landed in — a 50-token request can wait 40 seconds behind a stranger. That is a queueing pathology you have seen before: it is exactly a single-threaded worker pool where one slow job blocks the queue, and the fix is the same shape — decouple the scheduling unit from the job unit.

**⚠ Trap:** "we'll fix it by batching requests of similar expected length together." You cannot, because **you do not know the output length in advance** — that is the defining property of autoregressive generation. Length prediction is an active research area, not a production primitive, and any bucketing scheme you build on a predicted length degrades to static batching exactly when the prediction is wrong, which is exactly on the tail you were trying to protect. I would push back hard on this design in review.

### Walk me through one scheduler step in a modern engine, from the top.

I will describe it as the loop it is, because it is a loop and it runs 30–100 times a second.

**Phase 1 — reap.** Sequences that emitted EOS, hit `max_tokens`, or matched a stop string last iteration are removed from the running set and their KV blocks are decremented/freed. Their final tokens are handed to the detokenizer and streamed out. This happens *first* so the freed capacity is available to the same step's admission decision.

**Phase 2 — ensure the running set can take one more token.** For each running sequence, check whether `total_len + 1` crosses a block boundary and, if so, whether a free block exists. If capacity is insufficient, preempt victims (LIFO — newest first) until it is, freeing their blocks and pushing them back to the front of the waiting queue.

**Phase 3 — admit.** Walk the waiting queue in FIFO order. For each candidate, check both budgets: does adding it exceed `max_num_seqs` (the sequence-count budget), and does its prefill exceed the remaining `max_num_batched_tokens` (the token budget for this step)? Also check that its prompt's blocks can be allocated, minus whatever the prefix cache already gives you for free. Stop at the first request that does not fit — do **not** skip ahead to a smaller one, because that starves long prompts indefinitely.

**Phase 4 — build the batch metadata.** Concatenate the token ids of all prefilling sequences plus one token each for all decoding sequences into a flat 1-D tensor, build `cu_seqlens` (cumulative sequence lengths, the varlen convention), upload the block tables, upload slot mappings telling the kernel where each new K/V goes.

**Phase 5 — forward.** One model invocation. Prefill tokens and decode tokens go through the same linear layers as one big GEMM (this is the "mixed batch" that chunked prefill generalizes); attention runs with per-sequence extents.

**Phase 6 — sample and append.** Sample one token per sequence, write the new K/V into the slots, append the token, and loop.

The thing to notice, and to say: **phases 1–4 and 6 are CPU work on the critical path of a GPU loop.** If the forward pass is 20 ms and your Python scheduler takes 6 ms, you have lost 23% of your throughput to the host. That single observation is what drove the vLLM V1 rearchitecture.

### What are `max_num_seqs` and `max_num_batched_tokens` actually controlling, and draw me the latency/throughput curve as you turn each one.

They are the two admission budgets, and they control different axes, which is why people misconfigure them.

`max_num_seqs` caps **how many sequences are in the running set**. It is the concurrency knob. It bounds KV pressure (each sequence holds blocks) and it bounds the batch dimension of your decode GEMMs.

`max_num_batched_tokens` caps **how many tokens the model processes in a single forward pass**. It is the per-step work knob. A decoding sequence contributes 1 token; a prefilling sequence contributes its whole prompt (or its chunk, under chunked prefill). It bounds activation memory and it bounds how long any single iteration can take.

The curve, drawn with throughput on y and `max_num_seqs` on x: it rises steeply and nearly linearly at first, because decode is memory-bandwidth-bound — you are reading the same 70 GB of weights per step whether the batch is 1 or 64, so each added sequence is nearly free. Then it bends over at the **roofline knee**, where the batch has enough arithmetic intensity that you become compute-bound and additional sequences start costing real time. Past the knee it flattens, and eventually *declines* as KV pressure forces preemption and you spend cycles re-prefilling evicted work.

Overlay per-token latency (ITL) on the same x-axis: flat and low up to the knee, then rising linearly after it. **The knee is where you set `max_num_seqs`.** For a 70B on H100-class hardware that knee is typically somewhere in the 64–256 range depending on context length; you find it by sweeping, not by guessing.

The `max_num_batched_tokens` curve is different in character: it is a **TTFT-versus-ITL** dial, not a throughput dial. Large values let one big prefill run in a single step, which is great for that request's TTFT and terrible for everyone else's ITL because the whole batch is stalled for the duration of the prefill. Small values chunk the prefill across steps, keeping ITL smooth and pushing TTFT up.

**📐 Numbers you must know:** a rough starting point is `max_num_batched_tokens` in the 2,048–8,192 range for latency-sensitive interactive traffic and 16k+ for throughput-oriented batch traffic. **📅 Volatile:** engine defaults for both knobs have changed across vLLM versions (V1 raised several defaults and enables chunked prefill by default); check `EngineArgs` for the version you deploy rather than quoting a number from memory.

**⚠ Trap:** raising `max_num_seqs` to "get more throughput" without checking KV capacity. If `max_num_seqs × avg_tokens_per_seq × bytes_per_token` exceeds your KV pool, you have not increased concurrency — you have configured a preemption thrash loop, and throughput goes *down* because every preempted sequence re-runs its prefill. The rule I enforce: `max_num_seqs` must be derived from the KV budget and the measured length distribution, then reduced to the roofline knee — never set from a round number someone liked.

### Preemption: swap versus recompute. When do you pick which, and what's the modern default?

The decision is a straight bandwidth-versus-FLOPs comparison and you can do it on a napkin.

**Swap** moves the victim's KV to pinned host memory over PCIe and back. Cost = `2 × kv_bytes / pcie_bandwidth`. PCIe Gen5 x16 gives ~64 GB/s theoretical, ~50 GB/s achieved with pinned memory. For a 4,000-token 70B sequence: KV = 4,000 × 0.3125 MB = 1.25 GB, so 2 × 1.25/50 = **50 ms** round trip, plus it consumes PCIe bandwidth you may want for something else.

**Recompute** discards the KV and re-runs prefill on resume. Cost ≈ `2 × N_params × tokens` FLOPs = 2 × 70e9 × 4,000 = 5.6e14 FLOPs. At ~1,200 aggregate effective TFLOP/s on a 4-way H100 node in bf16 that is **~0.47 ms of pure compute**.

Recompute wins by two orders of magnitude on that arithmetic, and it wins even harder when prefix caching is enabled, because the prompt's blocks may still be cached and only the generated tail needs recomputing. The crossover where swap starts to win is where prefill compute becomes the scarce resource rather than PCIe: very long sequences on a heavily prefill-saturated engine, where recompute's FLOPs contend with incoming requests' TTFT.

But the number above understates recompute's real cost, and you should say so. Recompute's FLOPs do not execute in a vacuum — they consume `max_num_batched_tokens` budget in some future step, delaying admission of new requests. So the honest framing is: **recompute converts a latency cost into a scheduling cost**, and scheduling costs are easier to bound and easier to observe.

**📅 Volatile:** vLLM's V0 engine exposed both `RECOMPUTE` and `SWAP` preemption modes with `swap_space` configurable in GiB; the V1 engine consolidated toward recompute. Verify against your deployed version before asserting which modes exist.

**⚠ Trap:** budgeting host RAM for `swap_space` and then discovering it is not pinned. Pageable host memory forces the driver into a staging-buffer bounce that roughly halves effective transfer rate and adds jitter. If you enable swap you must budget *pinned* host memory, and pinned memory is not swappable by the OS — which means you can push your node into an OOM-killer situation on the host side while the GPU looks fine.

### How do you stop a single whale request from starving everyone, and how do you stop a long request from being starved forever?

Those are the two directions of the same fairness question and they need different answers.

**A whale hogging the engine** — one request with a 100,000-token prompt or a 32,000-token generation. The prompt side is handled by **chunked prefill**: the prefill is sliced to fit the per-step token budget, so it consumes a bounded share of each iteration rather than monopolizing one. The generation side is handled by the fact that decode is inherently fair — every running sequence gets exactly one token per iteration, so a whale gets 1/N of the batch's decode capacity, same as everyone. What it *does* hog is KV: 32k tokens × 0.3125 MB = 10 GB of the pool, which at 156 GB of KV is 6.4% of your entire capacity for one user. That is a **quota** problem, not a scheduling problem, and I solve it at the gateway with per-tenant token budgets, not inside the engine.

**A long request being starved** — the classic case is a 100k-token prompt that can never be admitted because the token budget is always partly consumed by small requests arriving continuously. This is why the admission loop must be **strict FIFO with head-of-line stop**: when the head of the waiting queue does not fit this step, you stop admitting, you do not skip to the next smaller request. Skipping ahead maximizes instantaneous utilization and guarantees indefinite starvation of large requests — a textbook livelock. Yielding a few percent of utilization to preserve FIFO is the correct trade, and it is the same reason you do not let a reader-preferring lock starve writers.

Priority classes sit on top: separate waiting queues per class (interactive / batch / background-agent), each with its own share of admission slots, so a batch job cannot consume interactive capacity. **⚠ Trap:** implementing priority as a strict preemptive ordering. Strict priority starves the low class under sustained high-class load, and low-class requests in an LLM system are usually *long* ones (batch evaluation, document processing), so you have re-created the starvation you just fixed. Use weighted shares with an aging term, not strict priority.

**🗣 Say this in the room:** "Admission has to be FIFO with a hard stop at the head, even though skipping ahead would raise instantaneous utilization — otherwise long prompts starve forever. Fairness between tenants belongs at the gateway as a token quota, not in the engine scheduler, because the engine has no idea who is paying you."

### 🏋 Drill: write me a toy continuous-batching scheduler in Python. Forty minutes, no libraries.

**Pass criterion:** in 40 minutes, unaided, produce a scheduler that (a) admits under both a sequence budget and a token budget, (b) allocates and frees KV in fixed-size blocks, (c) preempts LIFO when blocks run out, (d) reaps finished sequences at the top of the step, and (e) never skips ahead in the waiting queue. If it runs and the invariant `free_blocks + sum(len(r.blocks)) == total_blocks` holds after every step, you pass.

```python
from collections import deque
from dataclasses import dataclass, field

BLOCK = 16
ceil_div = lambda a, b: -(-a // b)

@dataclass
class Req:
    rid: int
    prompt_len: int
    max_new: int
    generated: int = 0
    blocks: list = field(default_factory=list)   # physical ids; None = opaque
    def total_len(self):     return self.prompt_len + self.generated
    def blocks_needed(self): return ceil_div(self.total_len() + 1, BLOCK)

class Scheduler:
    def __init__(self, total_blocks, max_num_seqs, max_batched_tokens):
        self.total, self.free = total_blocks, total_blocks
        self.max_seqs, self.tok_budget = max_num_seqs, max_batched_tokens
        self.waiting, self.running = deque(), []
        self.preemptions = 0

    def add(self, r): self.waiting.append(r)

    def _alloc(self, r, n):
        assert n <= self.free
        self.free -= n; r.blocks += [object()] * n

    def _release(self, r):
        self.free += len(r.blocks); r.blocks.clear()

    def _preempt(self, r):                        # recompute-style: drop all KV
        self.running.remove(r); self._release(r)
        r.generated = 0                           # will re-prefill on resume
        self.waiting.appendleft(r); self.preemptions += 1

    def step(self):
        # (1) reap finished sequences FIRST so their blocks fund this step
        for r in [x for x in self.running if x.generated >= x.max_new]:
            self.running.remove(r); self._release(r)

        # (2) make room for one more token per running sequence, LIFO preempt
        for r in list(self.running):
            if r not in self.running: continue
            need = r.blocks_needed() - len(r.blocks)
            while need > self.free and self.running:
                self._preempt(self.running[-1])
            if r in self.running and need > 0:
                self._alloc(r, need)

        # (3) admit under BOTH budgets; stop at the head, never skip ahead
        budget = self.tok_budget - len(self.running)      # decodes cost 1 token
        while self.waiting and len(self.running) < self.max_seqs:
            r = self.waiting[0]
            need = ceil_div(r.prompt_len, BLOCK)
            if r.prompt_len > budget or need > self.free:
                break
            self.waiting.popleft(); self._alloc(r, need)
            budget -= r.prompt_len; self.running.append(r)

        # (4) "run the model": every running sequence emits exactly one token
        for r in self.running:
            r.generated += 1
        assert self.free + sum(len(x.blocks) for x in self.running) == self.total
        return [r.rid for r in self.running]
```

Three things a strong candidate adds unprompted. First, the reap-before-admit ordering in phase 1 — get it backwards and you under-admit by exactly one step's worth of freed capacity every iteration. Second, the `break` rather than `continue` in phase 3, which is the starvation fix. Third, the assertion — a block-accounting invariant checked every step is how you catch the refcount bugs that otherwise present as silent cross-request corruption.

**⚠ Trap:** the common implementation error is preempting the sequence that *needs* the block rather than the newest one. Preempting the needy sequence produces a livelock — it goes back to the queue, gets readmitted, needs a block, gets preempted. LIFO victim selection is what makes the loop terminate.

### Explain the vLLM V1 engine architecture and what problem the rewrite was solving.

The problem was that a Python-hosted inference server has substantial **CPU work on the GPU's critical path**, and as GPUs got faster that CPU work went from noise to the dominant overhead. Concretely: HTTP handling and JSON parsing, tokenization, the scheduler's own bookkeeping, block-table construction, sampling parameter processing, detokenization, and streaming responses back — all of it in Python, all of it between forward passes.

Do the arithmetic. If a decode step for a 70B is 20 ms, 5 ms of Python is 20% overhead. Now put an 8B on the same card: the step is ~4 ms, and the same 5 ms of Python is **more than half your wall clock**. The engine is not GPU-bound any more, it is interpreter-bound, and no kernel optimization will help.

V1's answer is a process split. The **API server process** owns HTTP, tokenization, detokenization and response streaming. The **EngineCore process** owns the scheduler, the block manager and the model executor, and communicates with the API server over IPC with a compact message format. Because they are separate OS processes, the API server's per-token detokenization and socket writes for request A overlap with the GPU forward pass for request B, instead of serializing behind the GIL. That is the whole idea, and you already know why it needs processes rather than threads.

On top of that, V1 pushed toward **zero-overhead scheduling**: overlapping the CPU-side preparation of step `t+1` with the GPU execution of step `t`, so the scheduler's decisions are computed while the GPU is busy rather than while it is idle. It also made **prefix caching on by default** — the claim being that the hashing and lookup cost is small enough that there is no longer a reason to leave it off — and simplified the scheduler by removing the separate prefill/decode phase distinction, treating every step as a mixed token budget.

**📅 Volatile:** vLLM V1 details (default flags, which preemption modes survive, exact IPC design) have moved across releases. Describe the *architecture* — process split, overlapped scheduling, unified token budget — and say "verify the current defaults" rather than reciting a flag list.

**🗣 Say this in the room:** "V1 is a GIL argument, not a CUDA argument. Tokenization, detokenization and HTTP were serializing against the scheduler in one Python process, and on a small model that host work is over half the step time. Splitting the API server from the engine core lets host work for one request overlap the GPU pass for another. It's the same reason you'd never put your event loop and your CPU-bound serializer on the same core."

### Why does the scheduler need to know about the block manager? Couldn't you keep them separate?

You could, and the result would be an engine that OOMs or thrashes. The scheduler and the block manager are co-designed because **admission is a memory decision, not a queueing decision.**

Consider what "admit this request" means. It means: over the next `max_tokens` iterations, this sequence will consume up to `ceil((prompt + max_tokens)/16)` blocks, monotonically increasing, and it cannot be rolled back cheaply once it has started producing user-visible streamed output. A scheduler that admits on sequence count alone will happily admit 256 requests whose combined worst-case footprint is 3× the pool, and then discover the problem 400 iterations later — at which point the only remedy is mass preemption, which throws away 400 iterations of work.

So the scheduler asks the block manager three distinct questions every step, and each one is a different query: *can this waiting request's prompt be allocated now* (accounting for prefix-cache hits, which make some prompts nearly free); *can every running sequence take one more token*; and *if not, which victim frees the most blocks*. None of those are answerable from queue state.

The deeper coupling is prefix caching. A request whose 6,000-token system prompt is already resident costs 0 new blocks for that span and skips 6,000 tokens of prefill compute. That means **admission order changes admission cost** — admitting a cache-hitting request is cheaper than admitting a cache-missing one, so a scheduler that is blind to the cache makes systematically worse decisions. This is also why prefix-affinity routing at the load balancer interacts with engine scheduling: the router's placement decision changes the engine's admission economics.

**⚠ Trap:** the clean-architecture instinct says put a repository interface between them. Resist it here. The scheduler needs *speculative* queries ("what if I admitted this?") and needs them to be free, because it asks them 256 times per step at 50 steps per second — 12,800 queries per second. An abstraction that allocates a Python object per query costs you real throughput. This is one of the few places where I would accept tight coupling in review, and I would say so explicitly.

### What's the difference between throughput and goodput here, and why does it change how you set your knobs?

Throughput is tokens per second out of the engine. **Goodput is tokens per second out of the engine that were delivered inside their SLO.** A token that arrives after your p95 TTFT budget blew is, from the product's point of view, a token you paid for and cannot sell.

The distinction matters because the two are optimized by *opposite* knob settings. Throughput is maximized by cranking `max_num_seqs` to the roofline knee and beyond, running huge prefill chunks, and letting the queue absorb bursts — every one of which raises tail latency. Goodput is maximized by admitting fewer sequences than capacity allows, chunking prefill aggressively, and **shedding load** rather than queueing it.

Work an example. Suppose at `max_num_seqs = 64` you get 8,000 tok/s with p95 TTFT of 350 ms, and at `max_num_seqs = 256` you get 14,000 tok/s with p95 TTFT of 2,100 ms. If your product SLO is p95 TTFT ≤ 500 ms, the second configuration has **goodput of roughly zero above the 5th percentile of latency** — you are producing 75% more tokens and violating the contract on most of them. The naive throughput benchmark says config 2 is 1.75× better; the goodput measure says config 1 is the only shippable one.

**💰 Math:** put money on it. At $2.50/GPU-hour on a 4-GPU node the node costs $10/hour. Config 1 produces 8,000 × 3,600 = 28.8M tokens/hour → $10/28.8M = **$0.347 per million output tokens**. Config 2 produces 14,000 × 3,600 = 50.4M tokens/hour → $10/50.4M = **$0.198 per million** — 43% cheaper *per token produced*. But if the SLO breach causes 20% of sessions to be abandoned, your cost per *completed session* is higher under config 2. Cost-per-token is the wrong denominator; cost per **resolved task inside SLO** is the right one.

**🗣 Say this in the room:** "I don't tune for throughput, I tune for goodput — tokens delivered inside the latency SLO. They point in opposite directions: throughput wants a deep queue, goodput wants admission control and load shedding. If a benchmark reports tokens/sec without reporting the concurrency and the latency distribution it was measured at, it's not a number I can use."

### Where does host CPU time actually go in an inference server, and how would you find out?

Six places, roughly in order of how often they surprise people.

**Detokenization.** Every streamed token goes through incremental detokenization, which for BPE means maintaining partial-UTF-8 state per sequence and doing a string operation per token per sequence. At 256 concurrent sequences and 50 steps/s that is 12,800 detokenization calls per second, in Python.

**Sampling parameter processing.** Building the logit-processor stack (temperature, top-p, top-k, repetition penalty, logit bias, structured-output masks) per request per step. Naively implemented this is a per-sequence Python loop over a 128k-entry vocabulary tensor.

**Scheduler bookkeeping.** Block table construction and upload, slot mapping, sequence group metadata — all Python object churn, which is also GC pressure.

**Tokenization of incoming prompts.** Usually fast (Rust-backed HuggingFace tokenizers) but it happens on the request path and it is `O(prompt_len)`, so a 100k-token prompt is a real hit.

**HTTP/serialization.** JSON parse in, SSE frames out, one flush per token per stream.

**Structured-output mask computation.** If you are using grammar-constrained decoding, computing the allowed-token mask can be tens of microseconds per sequence per step, and it is on the critical path unless the engine overlaps it.

To find out, do not guess. `py-spy dump --pid <engine>` and `py-spy record` on the engine process give you a flame graph without instrumenting anything, and they work on a production pod. Then `nsys profile` on the same process, viewed as a timeline, shows you the thing that actually matters: **gaps between kernel launches**. If the GPU timeline has holes and the CPU timeline has Python frames filling them, you are host-bound, and no amount of kernel tuning will help.

**📐 Numbers you must know:** the diagnostic ratio is *GPU busy time ÷ wall clock* for the engine loop. Above ~90% you are GPU-bound and should be optimizing kernels or batch size. Below ~70% you are host-bound and should be optimizing Python, process layout, or CUDA graph usage. I have never seen a team that measured this ratio *first* waste a week on the wrong layer, and I have seen several that did not.

### An interviewer says "we serve an 8B model and our GPU utilization is 45% at max concurrency." What's your first hypothesis?

Host-bound, not GPU-bound — and specifically **kernel launch overhead plus Python scheduling**, because those costs are per-step constants while the model's work shrinks with model size.

The arithmetic makes it obvious. An 8B in bf16 is 16 GB of weights; on an H100 at 3.35 TB/s a decode step's weight read is 16/3350 = **4.8 ms** at 100% memory-bandwidth utilization, realistically ~7 ms at a healthy 70% MBU. Now count kernel launches: a 32-layer model issues on the order of 10–20 kernels per layer once you count the norms, projections, rotary embedding, attention, the MLP's three GEMMs and the activation — call it 400–600 launches per forward pass. At ~5 µs of CPU launch overhead each, that is 400 × 5 µs = **2 ms of pure launch cost**, or roughly 30% of a 7 ms step. Add 3–5 ms of Python scheduling and detokenization and you are at 45% utilization exactly as described.

So my ordered hypotheses: (1) **CUDA graphs are disabled** — someone set `--enforce-eager` to debug something in March and it never came back. That single flag typically explains most of it on a small model. (2) The API server and engine are in one process, serializing host work against the GPU loop. (3) Sampling or structured-output mask computation is running unbatched in Python. (4) Only then: batch size is below the roofline knee and you are genuinely bandwidth-starved, in which case utilization *should* be low and the fix is more concurrency, not less overhead.

The first thing I would actually run is `nsys profile` for ten seconds and look at whether the CUDA stream has gaps. Gaps mean host-bound; no gaps at 45% "utilization" means you are misreading `nvidia-smi`, which reports the fraction of time *any* kernel was resident — not how well those kernels used the SMs.

**⚠ Trap:** `nvidia-smi`'s `utilization.gpu` is not a utilization metric in the sense you want. It is "percentage of sampling intervals during which at least one kernel was executing." A kernel using 3% of the SMs for the entire second reads as 100%. Never make a capacity decision from it; use `nsys`/`ncu` or the engine's own step-time metrics.
