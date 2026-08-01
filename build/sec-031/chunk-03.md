### Walk me through how a GPU actually gets attached to a pod on Kubernetes. What's in the path?

Kubernetes has no native concept of a GPU. It knows about CPU and memory as first-class compressible/incompressible resources and about everything else as an opaque *extended resource*. The whole GPU story is a plugin that teaches the kubelet to advertise and allocate one.

The path, end to end:

1. **Driver on the host.** The NVIDIA kernel driver and userspace libraries must be present on the node. In a managed cluster this is a node-image concern; on your own nodes the **GPU Operator** installs and version-manages it, along with everything below.
2. **Container runtime hook.** The NVIDIA container toolkit installs a runtime (or an OCI prestart hook) that, when a container is created with the right annotations, bind-mounts the driver libraries and the `/dev/nvidia*` device nodes into the container. Without this, `nvidia-smi` inside your pod prints nothing and CUDA fails to initialize — the classic first-day failure.
3. **Device plugin.** A DaemonSet registers with the kubelet over a Unix socket and reports "this node has 8 devices of resource `nvidia.com/gpu`." The kubelet then advertises `nvidia.com/gpu: 8` in the node's allocatable resources.
4. **Scheduling.** Your pod requests `resources.limits["nvidia.com/gpu"]: 8`. The scheduler treats this as an integer extended resource: **it is not divisible and not overcommittable**. You cannot request 0.5 GPUs. Requests and limits must be equal.
5. **Allocation.** The kubelet asks the device plugin to allocate specific device IDs, and the plugin returns the environment (`NVIDIA_VISIBLE_DEVICES`) and device mounts for the container.
6. **Monitoring.** DCGM-exporter, another DaemonSet, scrapes per-GPU telemetry (utilization, memory, SM occupancy, temperature, XID errors) and exposes it to Prometheus. You want this on day one, and specifically you want **XID error alerts** — an XID 48 or 79 usually means a GPU is falling off the bus and the node needs to be cordoned.

**⚠ Trap:** treating GPUs like CPU in your resource model. GPUs are integral and non-overcommittable, which means every scheduling technique you rely on for CPU — burstable QoS, requests below limits, tight bin-packing — does not apply. A pod requesting 8 GPUs either finds a node with 8 free or pends forever. This changes cluster design: you want node pools shaped exactly like your workload's GPU count.

**⚠ Trap:** assuming a pod that requests 1 GPU is isolated from a pod on the same node requesting another. They share the PCIe bus, the NUMA node's memory bandwidth, and possibly the same NVLink domain. Two "isolated" pods on one node can absolutely contend, and if one of them is doing a large host-to-device weight load it will visibly slow the other's throughput.

### How do you lay out a mixed GPU cluster — node selectors, taints, priority classes? Give me the design.

The organizing principle: **GPUs are expensive and non-overcommittable, so the only thing that matters is that no GPU is ever occupied by something that did not need one, and no high-value workload is ever blocked by a low-value one.** Everything below serves those two invariants.

**Taints on every GPU node.** `nvidia.com/gpu=present:NoSchedule` on all GPU nodes, applied by the node pool config so it exists before the first pod schedules. Only pods with the matching toleration land there. Without this, your log shipper, your metrics agent and someone's CI job will schedule onto a $30/hr node and consume the CPU your inference server needed. This is the single highest-value line of YAML in a GPU cluster.

**Labels for the heterogeneity axes**, then node selectors or affinity against them: GPU model (`gpu.model=h100-sxm`), memory (`gpu.mem=80gb`), interconnect (`gpu.interconnect=nvlink`), and lifecycle (`capacity-type=spot|ondemand|reserved`). Do not encode "which team" — encode physical properties, and let workloads express requirements.

**Node pools shaped like the workload.** One pool of 8-GPU NVLink boxes for the big LLM; one pool of single-GPU L4/A10G nodes for embeddings and rerankers; optionally a pool of MIG-partitioned nodes for small-model and dev workloads. Do not run your embedding fleet on the 8-GPU pool — one embedding pod requesting 1 GPU on an 8-GPU node fragments the node so no 8-GPU inference pod can ever schedule there. That fragmentation is silent and expensive: you will see 7 idle GPUs and a pending pod.

**PriorityClasses with real semantics.** Something like: `llm-serving-critical` (highest, preemption-exempt), `llm-serving-standard`, `batch-inference` (preemptible), `experiment` (lowest, first to die). Set `preemptionPolicy: Never` on the low tiers so they cannot preempt anything, and let the high tiers preempt. This is what makes it safe to fill idle capacity with batch work — the batch pods evaporate when production needs the GPUs.

**Pod anti-affinity across replicas** so your three replicas of the same model do not all land on the same node or the same rack. A single node failure taking out your entire fleet is an outage you write a postmortem about.

**🗣 Say this in the room:** "Taint every GPU node so nothing schedules there by accident, label by physical property rather than by team, shape node pools to match the workload's GPU count so you don't fragment 8-GPU boxes, and use priority classes so batch work fills idle capacity and gets preempted the instant production needs it."

### What is gang scheduling and what specifically breaks without it?

Gang scheduling means a set of pods either all schedule together or none of them do. Kubernetes' default scheduler is pod-at-a-time and has no such concept, which produces a specific and vicious failure: **resource deadlock**.

Concretely. Two multi-node inference deployments each need 16 GPUs across two 8-GPU nodes, and the cluster has exactly 32 free GPUs on four nodes. The scheduler places job A's first pod, then job B's first pod, then A's second, then B's second… and in an unlucky interleaving each job holds 16 GPUs' worth of *placed but not-yet-runnable* pods while waiting for peers that can never be placed. Both jobs hang. Nothing errors. Your GPUs show as allocated and your workloads show as pending, and the only fix is a human deleting one of them.

The same thing happens with a Ray cluster that needs a head plus N workers before it can serve, or a multi-node vLLM deployment where the TP ranks must all exist before NCCL initialization completes. Partial placement produces pods sitting in a NCCL init timeout — which, by default, is long — burning GPU-hours doing nothing.

**Volcano** and **Kueue** are the two answers. Volcano is a full batch scheduler replacement with `PodGroup` semantics, `minAvailable`, queues, fair-share and preemption policies; it came out of the HPC/batch tradition. **Kueue** is the Kubernetes-SIG-native approach: it does not replace the scheduler, it *gates admission* — a Workload is only admitted (and its pods only unsuspended) when its full quota is available in a ClusterQueue, with cohorts for borrowing between teams. Kueue is the lighter-weight, more idiomatic choice and it composes with Jobs, JobSets and KubeRay. Volcano is the choice when you need richer intra-gang scheduling policies.

**⚠ Trap:** thinking gang scheduling is only a training concern. Any multi-node serving deployment — TP/PP across nodes, a Ray Serve cluster, a disaggregated prefill/decode pool — has the same all-or-nothing property. And the failure mode in serving is worse, because a half-placed serving deployment may pass its readiness probe on the pods that did schedule and start receiving traffic it cannot serve.

**⚠ Trap:** setting `minAvailable` equal to the *desired* replica count for an elastic workload. If your batch job can run at any world size ≥ 4, set minAvailable to 4 and let it grow. Setting it to 64 means the job waits for a 64-GPU window that may never open in a busy cluster, and your queue-wait metric quietly becomes hours.

### MIG, time-slicing and MPS all let multiple things share a GPU. Give me three workloads and tell me which you'd use for each.

They share a GPU in three genuinely different ways, and the difference is *isolation*.

**MIG (Multi-Instance GPU)** physically partitions the GPU at the hardware level: an A100 or H100 splits into up to seven instances, each with its own dedicated slice of SMs, its own slice of HBM, its own L2 and memory-bandwidth path. The partitions are hardware-isolated — one instance cannot affect another's performance or crash it. Each MIG instance appears to Kubernetes as a separate allocatable device (with profile names like `nvidia.com/mig-1g.10gb`). The cost is rigidity: reconfiguring the partition layout requires draining the GPU, and the profiles are fixed sizes, so you get fragmentation at the profile level.

**Time-slicing** is the device plugin advertising `nvidia.com/gpu: 4` when there is one physical GPU, and letting the driver context-switch between the four containers' CUDA contexts. There is **no memory isolation** — all four share the same 80 GB and one can OOM the others — and no performance isolation, since time-slicing serializes work rather than running it concurrently. It is oversubscription, plain and simple.

**MPS (Multi-Process Service)** runs client processes' kernels through a single CUDA context so they execute *concurrently* on the SMs rather than time-sliced. Better throughput than time-slicing for small kernels that individually cannot fill the GPU. Isolation is still weak: memory is shared (though MPS supports per-client memory limits), and a fault in one client historically could affect others.

Now the three workloads:

**Production LLM serving at scale** — none of the above. Give it whole GPUs. The model does not fit in a partition, and you want every SM and every byte of HBM for KV cache. Sharing here is a category error.

**A fleet of small models — a 1B classifier, a reranker, a guardrail model, each low-QPS** — MIG. You get hard isolation so the guardrail model's latency is unaffected by the classifier's traffic burst, and you turn one $30/hr H100 into seven independently-schedulable units. **💰 Math:** seven `1g.10gb` instances at, say, $2.50/GPU-hr means each effective unit costs $0.36/hr. Running those same seven models on seven whole L4s at $0.70/hr costs $4.90/hr versus $2.50/hr — MIG wins if you already have the H100s and the models fit in 10 GB, though a dedicated L4 pool is usually simpler and I would default to that unless the H100s are already there.

**Developer notebooks and CI inference jobs** — time-slicing. Isolation genuinely does not matter, utilization is near zero most of the time, and the goal is to let twenty engineers share four GPUs without a booking spreadsheet. Set a memory-limit convention and accept that someone will occasionally OOM a peer.

**⚠ Trap:** enabling time-slicing on production nodes to "improve utilization." Utilization looks better on the dashboard and p99 latency doubles, because your inference pod is now context-switching against someone's notebook. Utilization is not the objective; SLO attainment per dollar is.

### Our inference pod takes nine minutes from schedule to first token. Debug it.

Nine minutes is not one problem, it is four, and the discipline is to instrument each phase before touching anything. I want a timeline with these boundaries stamped: pod scheduled → image pulled → container started → weights loaded into host memory → weights on GPU → engine warmed (CUDA graphs / compile) → readiness probe passed.

The usual decomposition for a 70B:

**Node provisioning (0–4 min).** If the pod triggered a cluster-autoscaler scale-up, you paid for a cloud VM boot plus GPU driver init plus node registration. This is often the biggest single term and it is invisible if you only measure from container start. Fix: warm pools, or over-provisioning with low-priority placeholder pods that get preempted.

**Image pull (1–3 min).** A CUDA + PyTorch + vLLM image is easily 8–15 GB. At 200 MB/s from a registry that is 40–75 s, and worse if the registry is cross-region or you are pulling on many nodes at once. Fixes: pre-pull the image into the node image or via a DaemonSet, use a regional registry mirror, strip the image (multi-stage build, no dev toolchain, no duplicate CUDA copies), and consider lazy-pull/streaming image formats.

**Weight load (2–5 min) — usually the dominant term.** 140 GB from object storage. At 1 GB/s single-stream that is **140 seconds**; most naive downloads are far slower than that because they use one connection. With parallel multipart downloads on a high-network-bandwidth instance you might reach 3–5 GB/s, giving 28–47 s. Then you copy host→device over PCIe Gen5 at ~50 GB/s effective: 140/50 = 2.8 s, negligible by comparison. **Object storage bandwidth is the wall, and the fix is to not read from object storage.**

**Engine warm-up (0.5–2 min).** CUDA graph capture across the batch-size buckets, `torch.compile` if enabled, and any profiling run the engine does to size its KV pool. This is real and often ignored. TensorRT-LLM shifts it to build time instead.

**⚠ Trap:** the readiness probe passing before the engine is warm. The pod goes Ready, the Service adds it to endpoints, traffic arrives, and the first 30 requests each take 15 seconds while CUDA graphs capture. Your p99 spikes on every scale-up event and nobody connects the two. The readiness probe must exercise an actual generation, not a `/health` that returns 200 as soon as the HTTP server binds.

**🔍 Failure taxonomy — cold start, as a decision procedure:** measure the seven-boundary timeline first. If node provisioning dominates → warm pool or placeholder pods. If image pull dominates → pre-pull and slim the image. If weight load dominates → local NVMe cache or a PVC (next question). If warm-up dominates → capture fewer CUDA graph shapes, or accept it and fix the readiness probe. Do not optimize in a different order than the measurement tells you; I have watched a team spend a week slimming a container image when 70% of their cold start was the S3 read.

### Get me from a nine-minute cold start to under sixty seconds. What are the mechanisms?

Attack the weight-load term first, because it is the biggest and the fixes are well-understood.

**Local NVMe cache with a warming DaemonSet.** Most GPU instance types ship with several TB of local NVMe at 3–7 GB/s. Run a DaemonSet that pulls the model into `/mnt/nvme/models/<sha>/` on node startup, and mount that hostPath into inference pods. First pod on a fresh node still pays the download; every subsequent pod on that node reads at NVMe speed: 140 GB ÷ 5 GB/s = **28 s**. Key the directory by content hash so a model update is a new directory and rollback is instant.

**A ReadWriteMany PVC** (EFS/Filestore/a CSI-backed shared filesystem) holding the model, mounted read-only by all pods. Simpler than a DaemonSet, but network filesystem throughput is the risk — measure it, because many managed NFS tiers deliver only a few hundred MB/s per client and you have made things worse. I prefer NVMe + DaemonSet for large models and a PVC for small ones.

**Sidecar model-fetcher / init container** with parallel multipart download tuned for the object store: many concurrent range requests, a large part size, and enough threads to saturate the NIC. This is the difference between 200 MB/s and 3 GB/s and it is mostly a matter of configuration, not cleverness.

**Page cache.** If the model file was recently read on that node, it is in the host page cache and `mmap`-based loaders get it at memory speed. This is why the *second* pod on a node starts far faster than the first, and why your load test's cold-start numbers are optimistic if you keep restarting on the same node. Safetensors' zero-copy `mmap` load path is what makes this work — it is a real reason to prefer safetensors over pickle-based formats beyond the security argument.

**Weight streaming.** Rather than load-everything-then-serve, begin the forward pass as soon as layer 0 is resident and stream subsequent layers in behind the computation. Several platforms implement this (it is a headline feature of some managed providers). It can cut time-to-first-token dramatically on cold start, at the cost of a slow first few requests.

**Snapshot/restore.** Checkpoint the fully-initialized process — CUDA context, loaded weights, captured graphs — and restore it. CRIU-style checkpointing with CUDA support, or a VM/microVM snapshot. This is the only technique that also eliminates the engine warm-up term, and it is what gets you to single-digit seconds. It is also the most operationally fragile: driver version changes invalidate snapshots, and GPU-state checkpointing is still an area where tooling maturity varies. **📅 Volatile:** the maturity of CUDA checkpoint/restore has been improving quickly; verify current state before designing around it.

**📐 Numbers you must know for the cold-start budget:** object storage single-stream ~0.2–1 GB/s, parallel multipart 3–5 GB/s, local NVMe 3–7 GB/s, page cache ~10+ GB/s, PCIe Gen5 x16 ~50 GB/s, HBM3 3.35 TB/s. A 140 GB model crosses these in 140–700 s, 28–47 s, 20–47 s, ~14 s, and 2.8 s respectively. Every cold-start optimization is moving the model up that ladder.

### Warm pools versus scale-to-zero — how do you decide, and show me the money.

The decision is a straight comparison between the cost of idle GPUs and the cost of the latency (and lost requests) during a cold start, and it resolves differently for different traffic shapes.

**💰 Math, done concretely.** One 8×H100 replica at $2.50/GPU-hr = $20/hr = $14,600/month if it never sleeps. Suppose traffic is a business-hours product: real load 10 hours/day, 5 days/week = 50 hours/week out of 168, i.e. **30% duty cycle**. Scale-to-zero saves 70% × $14,600 = **$10,220/month per replica**. That is a real number and it is why people want it.

Now the cost side. With the optimizations above, cold start is 60 s. During those 60 s, requests either queue (users see a 60-second TTFT) or fail. If your traffic arrives as a step function at 9am, the first minute of every morning is degraded — annoying but survivable with a scheduled pre-warm. If your traffic is spiky and unpredictable all day, you will cold-start repeatedly and every spike's leading edge is a bad experience.

**The rule I use:**

- **Predictable diurnal traffic** → scheduled scaling, not scale-to-zero. Scale up at 08:30 on a cron, down at 19:00. You capture nearly all the savings with none of the cold-start risk, because you never cold-start on the user's request path.
- **Long-tail / rarely-used models** (per-customer fine-tunes, a specialty model used twice a day) → scale-to-zero is right, and you set customer expectations that the first request is slow. This is also where LoRA multiplexing beats separate deployments outright: fifty adapters on one always-warm base beats fifty scale-to-zero deployments.
- **Interactive production traffic with an SLO** → warm pool, always. Keep `N_min` replicas that never scale down, sized to your p10 traffic, and let the autoscaler add above that. The cost of an SLO miss on a paid product exceeds $14,600/month very quickly.
- **Batch/offline** → scale-to-zero aggressively, and use spot on top.

**⚠ Trap:** implementing scale-to-zero and forgetting the *scale-from-zero* path needs a component that survives at zero. If your only scaling signal is "queue depth on the inference server" and there is no inference server, nothing ever scales up. You need a request buffer or an external metric source (KEDA reading a queue, or an activator proxy that holds the request) that exists independently of the fleet. Knative's activator is the canonical shape of this. Teams discover this the hard way in staging, which is the good outcome.

### How do you version and roll back a 140 GB set of weights? Docker images aren't the answer, are they?

They are not, and treating them as one is a common mistake — a 140 GB layer in a container image makes every pull catastrophic, makes the registry expensive, and couples model rollout to code rollout, which is exactly the coupling you want to break.

The design I ship:

**Content-addressed storage, immutable.** Weights live in object storage under a path keyed by a digest: `s3://models/llama-3-70b-instruct/sha256-abc123.../`. The digest covers the config, tokenizer and all shards together, because a mismatched tokenizer is a silent quality bug that is miserable to diagnose. Nothing is ever overwritten in place.

**A registry that maps names to digests.** This can be MLflow's model registry, a Hugging Face private repo with revision pinning, or — my preference for a small team — a plain table (Postgres works fine) mapping `(model_name, stage) → digest`, plus metadata: training/quantization provenance, eval scores at promotion time, engine version compatibility, and who approved it. The registry is the source of truth; the deployment reads it.

**Deployment references a digest, never a tag.** The pod spec or a ConfigMap names `sha256-abc123`. A model rollout is a change to that reference, applied as a normal rollout. Rollback is setting it back to the previous digest — and because the previous weights are still in the NVMe cache on most nodes, rollback is *faster* than roll-forward, which is exactly the property you want in an incident.

**Eval gates as promotion criteria.** A digest cannot enter the `production` stage without a recorded eval run above threshold on your gold set, and that record lives with the registry entry. This is the thing that most distinguishes a mature setup, and it is where interviewers at applied-AI companies dig.

**⚠ Trap:** versioning weights but not the *serving configuration*. The same weights served with fp8 KV cache versus bf16 KV cache, or with a different chat template, or with `temperature` defaulted differently, are different products. Your deployable unit is `(weights digest, engine version, serving config, chat template, tokenizer)` and all five belong in the versioned artifact. I have seen a "weights rollback" fail to fix an incident because the actual regression was a chat-template change shipped in the same PR.

**🗣 Say this in the room:** "Weights are content-addressed artifacts in object storage, the registry maps a name and stage to a digest with the eval run that justified it, and the deployment references the digest. Rollback is a reference change, and it's fast because the old weights are still in the node-local cache. Code and weights ship on independent tracks."

### Walk me through canarying a new model version. What do you route on, what do you measure, and what triggers rollback?

The thing that makes this different from a normal service canary — and this is the point the interviewer is testing — is that **your primary signal is quality, quality is noisy, and you cannot detect a 3% quality regression from a 5-minute canary at 1% traffic.** Everything in the design flows from that.

**What I route on.** Deterministic hashing of a stable user or session identifier, not per-request randomness. Two reasons: a user must not see the old model on turn 1 and the new one on turn 2 of the same conversation (the prefix cache misses, and more importantly the persona shifts mid-conversation), and stable assignment is what makes the statistics valid. Percentage ramp: 1% → 5% → 25% → 50% → 100%, holding at each step long enough to accumulate signal.

**What I measure, in three tiers:**

*Tier 1 — instant, per-request, hard-fail signals.* Error rate, HTTP 5xx, tool-call schema validation failure rate, refusal rate, truncation/max-token-hit rate, output-length distribution. These move within minutes and any material shift is an automatic rollback. Output-length distribution is an underrated canary: a new model that suddenly writes 40% longer answers has changed your cost and your latency even if quality is fine.

*Tier 2 — fast operational signals.* TTFT p50/p95, inter-token latency, throughput per GPU, KV-cache hit rate, GPU memory headroom. A new version that is 15% slower per token is a capacity regression that will page you at the next traffic peak.

*Tier 3 — quality, which is slow.* Offline eval on the gold set must pass *before* the canary starts — that is a gate, not a canary metric. During the canary you run an LLM-judge comparison on a sample of live traffic (paired, same input, both versions, judged blind), plus product metrics: thumbs-up rate, copy/accept rate, task-completion rate, human escalation rate. These need days and thousands of samples, not minutes.

**Rollback triggers, stated as thresholds, automated:** error rate or schema-failure rate up more than 2× baseline over a 10-minute window; p95 TTFT up more than 25%; refusal rate up more than 50% relative; any Tier-1 signal outside its control band. Tier-3 regressions trigger a *hold*, not an automatic rollback, because the noise makes automation dangerous — a human reads the paired-judge results and decides.

**⚠ Trap:** canarying on aggregate quality and missing a segment. A new version can be net-positive overall and catastrophically worse on your Spanish-language traffic, or on your longest-context requests, or on one enterprise customer's document type. Slice every canary metric by language, context-length bucket, and top-10 tenants. The aggregate is a lagging, averaged indicator and it will hide the thing that generates the support ticket.

**⚠ Trap:** running a canary when the two versions have different prefix-cache behavior and comparing latency. A 1% canary has a cold prefix cache by construction, so its TTFT looks terrible for reasons that have nothing to do with the model. Either warm it, or exclude TTFT from the comparison until the canary is large enough to have a representative cache.

### Talk to me about GPU capacity reality — quotas, reservations, and what actually happens when you ask for 64 H100s.

This is the question that separates people who have run a GPU fleet from people who have read about one, and the honest answer is unglamorous: **capacity is a procurement problem before it is an engineering problem, and your architecture must assume you will not get what you asked for, when you asked for it.**

The layers of constraint, from softest to hardest:

**Account quota.** Your cloud account has a per-region, per-instance-family quota that often starts at zero for GPU families. Raising it is a support ticket with a human on the other end, it can take days to weeks, and the answer may be no. Ask early, ask for more than you need, and ask in more than one region.

**Actual physical availability.** Quota is permission, not inventory. An on-demand request for eight 8×H100 nodes in a popular region can simply return an insufficient-capacity error, repeatedly, for days. This is the part that surprises backend engineers most: you are used to `m5.xlarge` being effectively infinite.

**The commitment ladder**, roughly: on-demand (highest $/hr, no guarantee of availability), capacity reservations / capacity blocks (you pay for a window whether you use it or not, but the hardware is yours), committed-use or reserved instances (1–3 year commitment, 30–60% discount, and you are married to a GPU generation), and negotiated enterprise contracts. Neoclouds (CoreWeave, Lambda, Crusoe and similar) sit alongside these with different availability and pricing dynamics. **📅 Volatile:** discount levels, availability and the neocloud landscape all move fast — verify before quoting numbers.

**What this means architecturally**, which is the part an interviewer actually wants:

- **Be multi-region and multi-SKU capable.** Your deployment should be able to run on H100 *or* A100 *or* L40S with a different TP degree and a different throughput number, and your capacity planning should know all three numbers. A fleet that only runs on one SKU is a fleet that goes down when that SKU is unavailable.
- **Keep a managed-API fallback wired and tested.** When you cannot get hardware, you fail over to a provider at higher per-token cost rather than failing requests. Test the failover monthly; an untested failover is a comforting fiction.
- **Reserve your floor, burst on-demand.** Commit to the capacity your p10 traffic needs, so the base is cheap and guaranteed, and take the on-demand premium only on the peak.
- **Design for elastic degradation.** Under capacity shortage, route low-value traffic to a smaller model rather than dropping it. That routing tier is worth building before you need it.

**💰 Math:** committing to 16 H100s for a year at a 40% discount off $2.50/hr means $1.50/GPU-hr × 16 × 8,760 = **$210k/year** versus $350k at on-demand — a $140k saving, but only if utilization exceeds the break-even. Break-even utilization is 1.50/2.50 = **60%**: if you would have used the on-demand GPUs less than 60% of the hours, the commitment loses money. Compute that number before signing, and compute it against your *measured* diurnal curve, not your hoped-for one.

### Is spot or preemptible capacity ever right for inference? How would you make it safe?

For *interactive* inference, mostly no, and I would push back on it in a design review. For *batch* inference, yes, aggressively, and it is one of the biggest cost levers available.

The reason interactive is hard: preemption notice is short — on the order of 30 seconds to two minutes depending on cloud and instance type — and a 70B replica needs 60+ seconds just to load weights on a *replacement* node that may not exist. So a preemption means a real capacity gap, and if a meaningful fraction of your fleet is spot, correlated preemptions (the cloud reclaims a whole family at once) can take out several replicas simultaneously. The cost saving is real — spot commonly runs 60–80% below on-demand — but you are buying it with availability on the path that has an SLO.

If someone insists on spot for interactive serving, the safe shape is: **on-demand or reserved for the floor that covers p50 traffic, spot only for the burst tier above it, capped at some fraction (I use one third) of total capacity, with the router aware of which replicas are ephemeral and preferring stable ones for long-running or high-value requests.** Plus a hard rule that spot replicas are never the only holder of anything stateful.

For batch, the pattern is straightforward and it is the same one you already know from Celery/Kafka consumers, with one addition:

**Checkpointed drain.** Handle the preemption signal (a metadata-endpoint poll or a SIGTERM from a node-termination handler), immediately stop accepting new work items, finish or abandon in-flight items, **write progress to durable storage**, and exit cleanly. The critical design choice is that your unit of work is small enough to either complete or be safely abandoned within the notice window. If one work item takes 10 minutes and your notice is 30 seconds, you cannot drain — you can only lose work. So: chunk the batch into items of tens of seconds, checkpoint the completed-item set, and make re-processing idempotent via a deterministic output key (covered in the batch-pipeline questions).

**💰 Math:** a million-document extraction job needing 400 GPU-hours costs 400 × $2.50 = **$1,000** on-demand, or 400 × $0.75 = **$300** on spot at a 70% discount. Preemptions cost you re-work: if you checkpoint every 60 seconds and get preempted 20 times, you lose at most 20 minutes of GPU time = 0.33 GPU-hours ≈ $0.25. The re-work is noise; the saving is 70%. That asymmetry is why batch on spot is close to a free lunch and interactive on spot is not.

**⚠ Trap:** enabling spot and not testing the drain path. The drain handler is code that runs only during a rare event, so it is never exercised, so it is broken. Fire a synthetic preemption in staging on a schedule — the same discipline as chaos-testing a pod eviction — and assert that the checkpoint is written and the job resumes without duplicating output.

### How do you attribute GPU cost to teams or tenants when everyone shares one fleet?

The mental model: you cannot meter GPU *time* per request the way you meter CPU, because a request's GPU time is not separable — it was batched with fifty others. So you meter the thing that *is* attributable, which is **tokens**, and you convert tokens to dollars using a fleet-level rate you recompute periodically.

The mechanism:

**Emit a usage record for every request**, at the gateway, with: tenant/team ID, model, prompt tokens, cached prompt tokens, generated tokens, and the request's latency. This is a normal event stream — Kafka to a warehouse, or straight into Postgres if your volume is modest. Do this at the gateway rather than in the engine so it is uniform across self-hosted and API-backed models.

**Compute the fleet rate.** Every hour: total fleet cost (GPU-hours × $/hr, including idle) ÷ total tokens served in that hour = $/token, computed separately for prefill and decode tokens because their costs differ by an order of magnitude. Prefill tokens are cheap per token (compute-bound, huge batches); decode tokens are expensive (memory-bound). A blended rate is fine for chargeback and misleading for optimization, so I compute both.

**Charge idle time to someone deliberately.** This is the decision people avoid and then argue about at quarter-end. Idle capacity exists because of your SLO, not because of any one tenant. Either socialize it across tenants pro-rata by usage (simple, slightly unfair to small tenants) or hold it in a platform cost center and report it as the price of the SLO (my preference, because it makes the SLO's cost visible and negotiable).

**On Kubernetes**, the plumbing is: namespace per team, ResourceQuota on `requests.nvidia.com/gpu` per namespace so no team can exceed its allocation, labels on every pod carrying the cost-center, and a tool like OpenCost/Kubecost joining pod-hours to node prices. That gives you infrastructure-level attribution; the token records give you request-level attribution. You want both, and they should roughly reconcile — a large gap means someone is running work outside the gateway.

**💰 Math, worked:** fleet of 3 replicas × 8 H100s = 24 GPUs at $2.50/hr = $60/hr = $43,800/month. Measured monthly volume: 900M prefill tokens and 120M decode tokens. If you attribute cost by measured GPU-seconds (say prefill consumed 30% of fleet time and decode 70%), prefill rate = $13,140 / 900M = **$0.0146 per 1k prefill tokens**, decode rate = $30,660 / 120M = **$0.256 per 1k decode tokens** — a 17× ratio. Publish those two numbers internally and watch teams suddenly care about `max_tokens`. That behavioral effect is worth more than the accounting accuracy.

**⚠ Trap:** attributing purely on request count. One team sending 200 requests/day with 100k-token prompts costs far more than another sending 50,000 requests with 500-token prompts, and a per-request chargeback tells them the opposite. Tokens, split by phase, or you are actively misinforming your own organization.
