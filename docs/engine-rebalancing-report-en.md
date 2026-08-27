# Rollout Load Balancing in Dressage: From Sticky Session to Step Balance

The inference workload of agentic RL rollout is highly dynamic and long-tailed. Dressage's original sticky-session routing pins all generation steps of a session to the same SGLang Engine to maximize KV cache prefix reuse. This strategy provides good cache locality, but it does not reassign sessions as the actual load evolves during rollout. In the synchronous setting, once several long-tailed sessions concentrate on a few Engines, the completion time of the entire rollout batch is dictated by those Engines, while the remaining GPUs become idle early.

In a simplified estimate with 8 Engines and 128 trajectories, if the long tail grows the heaviest Engine's equivalent step count from 1,600 under ideal balance to 2,600, the synchronous rollout batch's completion time rises from 16s to 26s, cluster throughput drops from 800 step/s to about 492 step/s, and average GPU utilization falls from nearly 100% to about 61.5%. These numbers serve as a capacity-estimation example showing how Engine-level load skew translates directly into rollout time and GPU utilization losses.

Dressage Proxy Step Balance performs online scheduling at generation-step granularity on the Proxy side, choosing a target Engine based on Engine load snapshots, local scheduling deltas, and context recovery conditions, while a migration gain threshold suppresses unnecessary switches. The Proxy always sends the full `input_ids`, and the actual cache hit is still decided by SGLang, so its native cache correctness semantics are unchanged.

Across three workloads with different long-tail shapes, Step Balance improves Effective TPS/GPU, average GPU utilization, while reducing rollout time and GPU Spread, and the Coefficient of Variation (CV) of per-Engine step distribution drops from up to about **38%** to **below 2%**. The exact gain varies with the load skew of the workload; when the original assignment is already well balanced, the headroom for further improvement is relatively limited.

![Core performance metrics](../assets/step_balance/step-balance-main-metrics-en.png)

*Figure 1: Core performance metrics across the three multi-step workloads.*

## 01 Motivation: Why Step-Level Load Balancing

### The Inference Workload of Agentic RL

In agentic RL, a trajectory is usually not a single isolated model call, but a session composed of multiple generation steps. After the model produces an action, the agent may call tools, execute code, or interact with an external environment, and then issue the next generation carrying the previous dialogue and environmental feedback. The input of a later step typically contains the full context of all previous steps, so there are strong prefix-reuse opportunities within a session.

At the same time, the workload of different sessions is hard to predict accurately at the start:

- **Different numbers of interaction rounds**: simple tasks may finish quickly, while complex tasks produce more intermediate turns and generation steps.
- **Different context lengths**: tool outputs, retrieval results, code diffs, and execution logs keep expanding the prompt of some sessions.
- **Different generation lengths**: the number of response tokens varies across steps, with different effects on decode time and resource usage.
- **Different request re-entry times**: tool execution and sandbox operations desynchronize request arrivals across sessions within the same batch.

Agentic RL therefore has two seemingly contradictory scheduling requirements: consecutive steps of the same session should stay on the original Engine to reuse KV cache, yet the load across Engines keeps drifting as trajectories unfold. Even if round-robin assigns sessions evenly at the start, long trajectories and multi-step sessions can still generate more computation and queuing later, causing runtime imbalance.

### Sticky Session and KV Reuse

Dressage's original design uses session-level sticky routing: the Proxy passes `session_id` as the `routing_key` to the SGLang Router via `X-SMG-Routing-Key`; under the `consistent_hashing` policy, the generation steps of the same session continue to be routed to the same Engine.

The core benefit of this design is KV cache reuse. A later step of an agentic session often just appends a small number of tokens to the existing context; when the same Engine keeps serving it, the Engine can reuse the resident prefix KV and only compute the new suffix. Even if part of the prefix is evicted from GPU KV cache, the HiCache L2 in the deployment can retain a recoverable prefix in the same Engine's host memory, further strengthening cache locality within one Engine. If a step were arbitrarily switched to another Engine, the target Engine might have to re-prefill the full context, and the queuing gain from load balancing could easily be offset by the KV rebuild cost.

Sticky session is therefore a reasonable default for agentic RL inference: it provides stable cache locality at very low routing cost and avoids pointless back-and-forth session migration across Engines. The real problem is not stickiness itself, but that once the affinity is fixed, later routing never readjusts to the actual workload that emerges as the trajectory unfolds.

### The Long-Tail Bottleneck of Synchronous Rollout

In synchronous rollout, the training loop must wait for all trajectories in the current batch to finish before moving to the next stage. Round-robin can give each Engine a similar number of sessions at batch start, but if long trajectories happen to concentrate on a few Engines, those Engines gradually become hotspots. Under the sticky session policy, lightly loaded Engines cannot take over the later steps arriving at the hotspot Engines, so the batch completion time is stretched by a few hotspot Engines.

We use a simplified model of one synchronous rollout batch to estimate the impact of the long tail. Suppose the system deploys 8 SGLang Engines and initially generates 128 trajectories, with each Engine receiving 16. For ease of calculation, we first convert generation steps of different lengths and costs into equivalent steps, and assume a single Engine's processing capacity is 100 step/s.

Under ideal balance, each Engine processes 1,600 equivalent steps, and the whole batch has 12,800 equivalent steps:

```math
T_{\mathrm{ideal}}
= \frac{1{,}600}{100}
= 16\ \mathrm{s}
```

```math
\mathrm{Throughput}_{\mathrm{ideal}}
= \frac{12{,}800}{16}
= 800\ \mathrm{step/s}
```

The long tail of agentic rollout gradually differentiates the number of subsequent steps across Engines. Suppose the same batch still has 12,800 steps in total, but the distribution becomes the one shown below:

![Long-tail engine workload](../assets/step_balance/long-tail-engine-workload.png)

*Figure 2: Simplified long-tail load estimate with 8 Engines, 128 trajectories, and 12,800 equivalent steps. 1,600 steps marks the ideal balanced load; the 2,600 steps on Engine 1 mark the tail hotspot.*

Since a synchronous batch must wait for the last Engine to finish, the tail hotspot stretches the wall time to:

```math
T_{\mathrm{tail}}
= \frac{2{,}600}{100}
= 26\ \mathrm{s}
```

Effective cluster throughput drops to:

```math
\mathrm{Throughput}_{\mathrm{tail}}
= \frac{12{,}800}{26}
\approx 492\ \mathrm{step/s}
```

Compared with the ideal balanced state, throughput falls by about:

```math
1 - \frac{492}{800}
\approx 38.5\%
```

Average GPU utilization can be approximated as:

```math
\mathrm{Utilization}_{\mathrm{tail}}
= \frac{12{,}800 / 100}{8 \times 26}
= \frac{128}{208}
\approx 61.5\%
```

This example shows that even with the same total amount of work, as long as the long tail concentrates on a few hotspot Engines, synchronous rollout is slowed by the tail completion time, the other GPUs become idle earlier, and overall utilization drops accordingly. However, if requests on heavily loaded Engines are migrated to lightly loaded ones as illustrated in the figure, the batch's tail completion time can still be effectively shortened and overall utilization improved, despite the additional migration overhead this may incur.

## 02 Problem Analysis: Why Sticky Session Struggles with Runtime Long Tails

### Fixed Session Affinity Cannot Adapt to Runtime Load Changes

The long-tail phenomenon described above essentially comes from the tension between fixed affinity and dynamic load. Sticky session decides Engine affinity at session creation or first request, but the load of agentic rollout only emerges gradually over many subsequent generation steps: the router sees a stable `session_id`, while what actually enters the Engine queues are batches of steps with different arrival times, context lengths, and output lengths. A balanced initial assignment therefore does not imply balanced runtime load.

This tension shows that the problem is not that sticky session is entirely wrong, but that it lacks a runtime correction mechanism. All generation requests pass through the Proxy, making it a natural point for step-level scheduling, so a more reasonable approach is to use each generation step as the scheduling unit: when a step arrives, the Proxy re-evaluates which Engine should take it based on the current load of each Engine, rather than simply reusing the session's initial affinity. Such online decisions have two prerequisites: load observation must be refreshed asynchronously in the background without blocking the request path, and the delay between observation and dispatch must be accounted for, so that many concurrent steps do not all rush to the same low-pressure Engine based on the same stale observation.

### Cross-Engine Step Migration Is Constrained by KV Recovery Cost

Once each step becomes a scheduling unit, the dominant cost is not routing-table updates but cross-Engine recovery of the KV cache. The historical prefix of an existing session has usually already materialized as KV on the owner Engine; if a later step is migrated to another Engine, the target Engine must be able to reuse that prefix, otherwise it has to re-prefill the full context.

This directly changes whether a migration is worthwhile. For short contexts or light steps, re-prefill may be cheap; but in long-context, multi-turn agentic sessions, a full prefill can be more expensive than queuing. In that case, simply picking "the Engine that currently looks emptiest" does not necessarily shorten end-to-end time, and may even turn a queuing problem into a recomputation problem.

Step Balance therefore cannot optimize load alone; while balancing load, it must also account for whether context recovery is supported. Only when the target Engine is compatible with the current session's cache fingerprint, and recovery links such as HiCache L3 and Mooncake can provide cross-Engine KV reuse, does a low-load Engine truly gain from taking over the current step. Conversely, migration must also be gated by gain: a migration is only worth executing when the load gain suffices to cover the KV recovery and extra prefill cost, so that minor pressure fluctuations do not bounce sessions back and forth between Engines.

The resulting design is to introduce step-level online rebalancing on the Proxy side, using current Engine pressure as the request-assignment signal and KV recovery capability plus correctness boundaries as hard constraints. This avoids both permanently pinning long-tail load onto a few Engines at session granularity, and introducing excessive context rebuild cost for the sake of balance.

## 03 Step Balance in Dressage

Dressage introduces Step Balance on top of sticky session: a step-level online assignment mechanism. When each generation step arrives, the Proxy no longer relies solely on the session's initial affinity. Instead, it combines Engine load snapshots, local reservations, and context recovery conditions to choose the Engine currently best suited to take the request.

### Enabling and Configuration

Step Balance is disabled by default. Add `--enable-engine-rebalancing` when starting the Dressage Proxy to enable it; the other two parameters control the migration threshold for existing sessions and the Engine load snapshot polling interval.

| CLI Argument | Default | Purpose |
|---|---:|---|
| `--enable-engine-rebalancing` | Disabled | Enables new-session placement, per-step migration of existing sessions, and failover |
| `--engine-rebalancing-min-load-improvement-ratio` | `0.10` | Requires at least a 10% Pressure improvement for voluntary migration of an existing session |
| `--engine-rebalancing-load-snapshot-poll-interval-ms` | `60` | Sets the `/v1/loads` polling interval for each Engine |

### Overall Idea: From Session to Step

Step Balance lives inside the Dressage Proxy. Upstream Agents send OpenAI-compatible chat completion requests to the Proxy; the Proxy makes the routing decision before forwarding to an SGLang Engine.

The key difference from sticky session is scheduling granularity. Sticky session fixes the affinity once at session creation and never adjusts it; Step Balance treats each generation step as the scheduling unit, making one online decision per step based on the latest usable load snapshot and local reservations, and that decision only affects the target Engine of the current step. Stickiness is not abandoned: the sticky owner still provides the baseline affinity and the local KV reuse path, existing sessions stay on the owner by default, and cross-Engine assignment only happens when the load gain is sufficient and the recovery conditions are met.

<div align="center">
  <img src="../assets/step_balance/step-balance-architecture.png" alt="Step Balance architecture" width="96%">
</div>

*Figure 3: Step Balance architecture.*

The architecture maintains three categories of core state:

- **Engine load snapshots.** A background control loop periodically reads each Engine's running requests, active tokens, queue, health, and version information. The request path never probes Engines one by one; it only reads the latest usable snapshot.
- **Local reservations.** Before dispatching a request, the Proxy records the local delta introduced by this scheduling decision, covering the visibility window between snapshot refreshes and the real Engine state.
- **Session and recovery state.** The Proxy records each session's current owner, historical tokens, and cache fingerprint, and, combined with the HiCache L3 and Mooncake recovery links, determines whether a cross-Engine migration meets the context recovery conditions.

A step goes through four stages from arrival to completion:

1. **Refresh snapshots in the background.** The control loop polls each Engine's load snapshot at a fixed interval (running requests, active tokens, queue, health, and version); the request path performs no synchronous probing and only reads the most recently published snapshot.
2. **Build candidates and estimate pressure.** When a step arrives, the scheduler first filters candidate Engines by health, version compatibility, snapshot freshness, and context recovery conditions, then merges the snapshot load with local reservation deltas not yet covered by the snapshot into an effective load, and adds the current step's delta to obtain the projected pressure.
3. **Select a target and write a reservation.** The scheduler picks the candidate with the lowest projected pressure: new sessions or mandatory failovers take it directly; an existing session leaves its current owner only when the relative gain exceeds the threshold. Once the target is decided, the Proxy writes a local reservation before forwarding the request, so subsequent concurrent steps immediately see the expected load of this decision and do not all rush to the same Engine that looked idle under an old snapshot.
4. **Complete and release the reservation.** When the request finishes or fails, the Proxy releases the corresponding reservation, preventing the local estimate from staying above the Engine's actual load for long; on success, the current target is committed as the session's new owner, against which later steps continue to be evaluated.

### Scheduling Model

In the implementation, Engine selection is not an unconstrained `argmin` over all workers. The model must first determine which Engines can take the current step, and then compare candidate Engines with a single set of pressure metrics. The goal is not to precisely predict the completion time of a single request, but to describe each Engine's current load and near-future load trend with signals that are stable enough and available online.

These signals come from three paths. Engine-side running requests, queued requests, active tokens, token capacity, request capacity, and `token_usage` are obtained by polling SGLang `/v1/loads` in the background; request-side `session_id`, full `input_ids`, prompt length, and the current step boundary are parsed by the Proxy at the OpenAI-compatible request entry; queue/token deltas not yet covered by the next snapshot come from the Proxy's local reservations. Session owner, historical committed tokens, cache fingerprint, and context recovery state are maintained by the Proxy's session and rebalancing state.

#### Candidate Engine Set

The candidate set first answers "which Engines are eligible to take this step". The scheduler filters the currently discovered Engines one by one:

- The Engine must be healthy, and its latest load snapshot must not exceed the stale threshold.
- The Engine's weight version must match the version expected by the current request.
- The Engine must report a valid request capacity and token capacity.
- For an existing session, the target Engine's cache fingerprint must be compatible with the current session, so the request is never sent to an Engine that cannot interpret the same KV layout.
- For voluntary migration away from a healthy owner, the target Engine must additionally be context recovery ready.

New sessions have no historical cache fingerprint constraint and can choose directly among healthy, version-compatible Engines. Mandatory failover first removes an unhealthy or version-incompatible owner and, when necessary, accepts a full prefill instead of waiting for the original owner to recover.

Concretely, context recovery ready here means three things: the source and target have compatible deployments, the Mooncake recovery link has been calibrated, and the target model's cache profile is available. Mandatory failover is exempt from this condition and falls back to a full prefill when necessary.

#### Effective Load and Future Deltas

The scheduler cannot look only at the load already observed by the Engine in the snapshot. Because snapshots refresh asynchronously, requests the Proxy has just dispatched may not yet appear in the next `/v1/loads` result. If these deltas were ignored, many concurrent steps would see the same low-pressure Engine at the same time and make the same choice.

The Proxy therefore merges the snapshot load with local scoring deltas not yet covered by the snapshot into an effective load. For a candidate Engine $e$, let $R_{\mathrm{run}}$ be the effective running requests, $N_{\mathrm{token}}$ the effective token occupation, and $Q_{\mathrm{wait}}$ the effective queued requests:

```math
\begin{aligned}
R_{\mathrm{run}}(e)
&= R_{\mathrm{snapshot}}(e),\\
N_{\mathrm{token}}(e)
&= N_{\mathrm{snapshot}}(e)
+\Delta N_{\mathrm{local}}(e),\\
Q_{\mathrm{wait}}(e)
&= Q_{\mathrm{snapshot}}(e)
+\Delta Q_{\mathrm{local}}(e).
\end{aligned}
```

Here $R_{\mathrm{snapshot}}$, $N_{\mathrm{snapshot}}$, and $Q_{\mathrm{snapshot}}$ correspond to `running`, `active_tokens`, and `queued` in the snapshot, while $\Delta N_{\mathrm{local}}$ and $\Delta Q_{\mathrm{local}}$ come from the Proxy's local reservations.

A snapshot poll records the current reservation revision before issuing the request. Only when that poll successfully publishes a new snapshot are the local deltas that existed before the poll acknowledged and removed from scoring; reservations created during the poll, and deltas from before a failed poll, are retained. Each delta therefore participates in scoring until a new snapshot is published and leaves scoring afterwards, avoiding both undercounting and double-counting against the snapshot.

#### Per-Step Deltas

The current snapshot alone is not enough; scheduling must also account for the extra pressure that assigning the current step to each Engine would create. Let $N_{\mathrm{prompt}}$ be the full input length of the current step, and $N_{\mathrm{lcp}}$ its longest common prefix length with the session's last committed tokens. Every candidate Engine first gains one queue item:

```math
\Delta Q_{\mathrm{step}}=1.
```

The scoring token delta has three cases depending on the step's relationship to the owner:

```math
\Delta N_{\mathrm{step}}=
\begin{cases}
\max(0,N_{\mathrm{prompt}}-N_{\mathrm{lcp}}),
& \text{stay on the current owner},\\
N_{\mathrm{prompt}},
& \text{voluntary migration to another Engine},\\
N_{\mathrm{prompt}},
& \text{new session or mandatory failover}.
\end{cases}
```

**Why voluntary migration is scored at the full prompt.** When a step stays on the owner, the historical prefix is very likely already present in local KV, so the additional capacity pressure is mostly the suffix. When the step migrates to another Engine, however, the amount of prefix the target can recover through Mooncake L3 depends on the cache state at recovery time: whether the prefix is still in L3, whether it has been evicted, and how much available space the target Engine has at that moment. These factors cannot be known precisely when the request enters the Proxy. If scoring deducted an "assumed recoverable" prefix up front, then whenever the actual hit falls short of the assumption, the Proxy would systematically overestimate the target Engine's free space and steer too many steps toward Engines that only look idle, causing over-aggressive migration. Scoring therefore compares candidates under the worst case, namely the capacity pressure of the full prompt. In other words, the scheduling score describes an upper bound on the target Engine's capacity pressure after taking over the current step, not the actual number of prefill tokens. This upper bound does not stay distorted for long: the Engine-reported `token_usage` folds real cache occupation into the pressure, so the actual occupation formed by recovered or shared KV on the target Engine is still reflected, and the target does not appear emptier than it really is.

Execution and observability still need a prefill estimate, so the lifecycle reservation separately maintains a page-aligned LCP. Let $N_{\mathrm{lcp,page}}$ be the LCP rounded down to the target Engine's KV page size, and $N_{\mathrm{prefill}}$ the execution-side estimated number of prefill tokens:

```math
N_{\mathrm{prefill}}=
\begin{cases}
\max(0,N_{\mathrm{prompt}}-N_{\mathrm{lcp}}),
& \text{stay on the current owner},\\
N_{\mathrm{prompt}}-N_{\mathrm{lcp,page}},
& \text{Mooncake-ready voluntary migration},\\
N_{\mathrm{prompt}},
& \text{new session, mandatory failover, or unrecoverable}.
\end{cases}
```

The deltas used for scheduling scores are recorded separately from the reservation over the request's lifecycle. The former contains only $\Delta Q_{\mathrm{step}}$ and $\Delta N_{\mathrm{step}}$ and is used to compare candidate Engines; the latter records the request count, $N_{\mathrm{prompt}}+N_{\mathrm{out}}$ reserved tokens, and $N_{\mathrm{prefill}}$, and is used for resource-occupation accounting while the request executes.

#### Projected Pressure

Projected pressure consists of request, token, and queue pressure, covering the Engine's running-request pressure, KV/token space pressure, and the queuing tail that has already formed. For each candidate Engine $e$, the scheduler normalizes the three terms and sums them into a projected score for the state after the target Engine takes the current step. Let $C_{\mathrm{req}}$ be the request capacity, $C_{\mathrm{token}}$ the token capacity, and $U_{\mathrm{token}}$ the Engine-reported `token_usage`:

```math
\begin{aligned}
P_{\mathrm{run}}(e)
&=
\frac{R_{\mathrm{run}}(e)}
{C_{\mathrm{req}}(e)},\\
P_{\mathrm{token}}(e)
&=
\max\left(
\frac{N_{\mathrm{token}}(e)+\Delta N_{\mathrm{step}}(e)}
{C_{\mathrm{token}}(e)},
U_{\mathrm{token}}(e)
\right),\\
P_{\mathrm{queue}}(e)
&=
\frac{Q_{\mathrm{wait}}(e)+\Delta Q_{\mathrm{step}}}
{C_{\mathrm{req}}(e)},\\
P_{\mathrm{total}}(e)
&=
P_{\mathrm{run}}(e)
+P_{\mathrm{token}}(e)
+P_{\mathrm{queue}}(e).
\end{aligned}
```

Both `request_capacity` and `token_capacity` come from the Engine snapshot. Capacity normalization lets Engines with different TP/DP topologies or cache sizes be compared on the same scale; `max(..., token_usage)` ensures the Proxy's token estimate never falls below the Engine's own reported usage. `token_usage` tracks real cache pressure more closely than a plain prompt-token estimate, because it describes the token capacity currently occupied by the Engine’s KV state after deducting available space and evicted/reclaimable cache.

#### Engine Selection

The scheduler first finds the Engine with the lowest projected pressure in the candidate set $\mathcal{C}$:

```math
e_{\mathrm{best}}
= \arg\min_{e\in\mathcal{C}}
P_{\mathrm{total}}(e).
```

The code treats candidates whose scores differ by no more than $\tau$ ($10^{-7}$, `SCORE_TOLERANCE` in the code) as tied. If the sticky owner is in the tied set, the scheduler prefers to keep the owner; otherwise it generates a stable ranking with `SHA256(session_id, engine_url)`, making decisions reproducible across runs while spreading different sessions steadily across tied Engines.

For an existing session with a healthy owner, the scheduler also computes the relative load improvement to avoid flip-flopping the assignment for marginal gains:

```math
G_{\mathrm{load}}
= \frac{
P_{\mathrm{total}}(e_{\mathrm{owner}})
-P_{\mathrm{total}}(e_{\mathrm{best}})
}
{\max(P_{\mathrm{total}}(e_{\mathrm{owner}}),\varepsilon)}.
```

Here $\varepsilon=10^{-9}$ is a small constant guarding against division by zero (`_RATIO_EPSILON` in the code). The final selection rule is:

```math
e_{\mathrm{target}}=
\begin{cases}
e_{\mathrm{best}},
& \text{new session or mandatory failover},\\
e_{\mathrm{best}},
& \text{existing session with }
e_{\mathrm{best}}\neq e_{\mathrm{owner}}
\text{ and }
G_{\mathrm{load}}\geq\theta_{\mathrm{min}},\\
e_{\mathrm{owner}},
& \text{otherwise}.
\end{cases}
```

$\theta_{\mathrm{min}}$ corresponds to `min_load_improvement_ratio` in the implementation, with a default of `0.10`. This threshold gates migration: small load fluctuations never trigger a migration, and only a sufficiently large expected improvement replaces the current owner. New sessions and mandatory failovers have no owner-gain threshold and directly take the lowest-pressure candidate. If the owner is healthy but its snapshot is unavailable, the system keeps the owner; if a failover or a new session has no fresh snapshot for the moment, the same stable hash ranking is used to pick a healthy Engine, rather than migrating aggressively on incomplete observations.

## 04 Evaluation

We compare sticky session with Step Balance under identical runtime configurations, and evaluate scheduling effectiveness in terms of rollout completion time, effective throughput per GPU, and cross-Engine load dispersion.

### Experimental Setup and Metrics

The experiments run on a single machine with 8 GPUs, using Qwen3.5-4B as the rollout model and deploying 8 single-GPU SGLang Engines. The experiments use synchronous rollout with batch size 256, one trajectory sampled per prompt, and sampling temperature 0. All experiments use the same model, data, random seed, and inference configuration to keep the results comparable.

| Setting | Value |
|---|---:|
| Model | Qwen3.5-4B |
| Rollout Engines | 8 (1 GPU / Engine) |
| Batch size | 256 |
| Context length | 256K tokens |
| Response length | 4K tokens |
| Sampling | temperature=0 |
| HiCache | ratio=2.0, `write_through`, `page_first_direct` |
| L3 cache | Mooncake (segment size=64 GB) |

The baseline uses sticky session, and the experimental group enables Step Balance under the same configuration. The load snapshot polling interval is set to 60 ms; the minimum migration gain threshold is configured separately for each workload distribution.

### Workload Configuration

We treat the workload distribution as an important experimental factor for scheduling performance. Under sticky session, different trajectory-length compositions produce different degrees and shapes of cross-Engine load skew. Based on load shapes repeatedly observed in internal stress tests, we abstract three workloads covering different trajectory-length compositions and skew patterns: "short tails mixed with a broad long tail", "many single-step trajectories mixed with a broad long tail", and "a broad short tail mixed with a concentrated long tail", corresponding to data distributions A, B, and C in the figure.

![Repeat multi-step workload distributions](../assets/step_balance/repeat-multistep-workload-distributions.png)

*Figure 4: Step distributions of the three multi-step workloads. The x-axis is the number of generation steps per session; the y-axis is the corresponding number of sessions.*

In the experiments, the minimum migration gain threshold is set to 0.0 for Dataset A and Dataset B, and to 0.1 for Dataset C.

Under these workloads, we further observe how Step Balance responds dynamically to the runtime load skew produced by sticky session. When both the load snapshots and local reservations indicate that an Engine stays hot, and a migration satisfies the KV recovery conditions and the minimum gain threshold, subsequent steps are reassigned to less loaded candidate Engines, shortening how long a local hotspot persists; when the current assignment is already well balanced, the scheduler keeps the original owner and triggers no extra migration, avoiding unnecessary KV recovery overhead.

### Performance and Load Balance

Step Balance shows positive gains on all three workloads, with observed Effective TPS/GPU improvements in the range of 39.4%–64.2%. The higher gains correspond to workloads with more pronounced long tails and load skew, and should not be directly extrapolated as a universal gain; when the original assignment is already well balanced, the headroom for further improvement is relatively limited. At the same time, average GPU utilization rises, rollout time and GPU Spread fall, and the CV of the per-Engine step distribution drops from up to about **38%** to **below 2%**.

![Core performance metrics](../assets/step_balance/step-balance-main-metrics-en.png)

*Figure 5: Core performance metrics across the three multi-step workloads.*

![Engine step distribution balancing effect](../assets/step_balance/step-balance-engine-steps-deviation-en.png)

*Figure 6: Deviation of each Engine's actual processed steps from the mean.*

The step distribution shows that under sticky mode, subsequent requests are not spread evenly across the 8 Engines: some Engines process noticeably more steps than average, while others become idle early. With Step Balance enabled, the step counts of all Engines return to around the mean, and the load gap between GPUs shrinks accordingly. By utilizing otherwise idle GPU capacity, the cluster processes more effective tokens per unit time, so Effective TPS/GPU rises and the rollout finishes earlier.

### Rollout Tail Utilization

The performance difference mainly appears in the rollout tail, not in the first half. Both modes keep all 8 GPUs nearly saturated at the beginning; as short trajectories finish one after another, sticky's remaining work gradually concentrates on a few Engines, while Step Balance can keep using the already-idle Engines for the later steps of long trajectories.

![Mean GPU utilization over time](../assets/step_balance/step-balance-cluster-tail-en.png)

*Figure 7: Cluster mean GPU utilization over time during rollout.*

![Per-Engine GPU utilization heatmap](../assets/step_balance/step-balance-gpu-heatmaps-en.png)

*Figure 8: Heatmap of per-Engine GPU utilization over normalized rollout progress. The 75% vertical line marks the start of the rollout tail.*

The time series further shows that the difference mainly appears in the second half of the rollout. Under sticky, some Engines finish early and go idle, and mean GPU utilization steps down accordingly; in the last 25% of the interval, the heatmap already shows large blank areas with only a few Engines still working. Step Balance assigns subsequent steps to these idle Engines, so high utilization lasts longer and the Engines' finish times are closer together. Taking Dataset B as an example, the overall completion time drops from **3.19 hours** to **2.08 hours**. These results show that the key to how Step Balance shortens rollout is not making the first half faster, but reducing tail waiting.

## 05 Future Work

### Current Scope: Single-Step Greedy

Step Balance currently schedules at single generation-step granularity. When a step arrives at the Proxy, the scheduler immediately makes a greedy choice based on the latest Engine snapshot and local deltas. This is the current default path, and will also serve as the fallback for future batch scheduling.

### Single-Step Greedy → Multi-Step MILP

**From 1-to-N to M-to-N.** The current per-step greedy approach suits low-latency online decisions, but it is essentially an online assignment of 1 request to N Engines: each step does its own `argmin`, each decision changes the load seen by later steps, the final distribution of the whole request group depends on arrival order, only per-step local optimality is guaranteed, and the overall gain and KV recovery cost of different migration combinations are never explicitly compared. The next phase introduces step-batch scheduling, extending the problem to a joint assignment of M requests to N Engines: the Proxy groups ready steps arriving within a short window into a small batch that shares one load snapshot, models the problem directly with Mixed-Integer Linear Programming (MILP), and minimizes the maximum Engine pressure after the batch is committed. Since any allocation produced by the greedy policy is a feasible solution of the MILP, its optimal value $z^*$ is no worse than per-step greedy—instead, it is globally optimal within the model given the snapshot and delta estimates; of course, this optimality is bounded by snapshot freshness and the fidelity of the delta estimates, and is not equivalent to the global optimum of the real system.

Let $\mathcal{B}$ be the set of steps in the current batch, $\mathcal C_s$ the candidate Engine set of step $s$, and $x_{s,i}\in\{0,1\}$ indicate whether step $s$ is assigned to Engine $i$. $L_i^{\mathrm{base}}$ is the Engine's current base pressure, and $\Delta L_{s,i}$ the additional load pressure if that Engine takes the step. Introducing an auxiliary variable $z$ for the maximum Engine pressure after the batch is committed, the master problem is:

```math
\begin{aligned}
\min_{x,z}\quad & z\\
\text{s.t.}\quad
& \sum_{i\in\mathcal{C}_s}x_{s,i}=1,
&& \forall s\in\mathcal{B},\\
& L_i^{\mathrm{base}}
+\sum_{\substack{s\in\mathcal{B}\\i\in\mathcal{C}_s}}
x_{s,i}\Delta L_{s,i}
\leq z,
&& \forall i,\\
& x_{s,i}\in\{0,1\},
&& \forall s\in\mathcal{B},\ i\in\mathcal{C}_s.
\end{aligned}
```

On top of this, $C_{s,i}^{\mathrm{migration}}$ can denote the number of KV tokens that must be restored or recomputed to migrate step $s$ to Engine $i$ (when the target can restore the prefix from shared L3, the cost mainly comes from restore plus the remaining prefill; otherwise it is close to a full prefill). After the maximum pressure reaches its optimal value $z^*$, keeping the uniqueness and binary constraints, we continue to solve:

```math
\begin{aligned}
\min_x\quad
& \sum_{s\in\mathcal{B}}\sum_{i\in\mathcal{C}_s}
x_{s,i}C_{s,i}^{\mathrm{migration}}\\
\text{s.t.}\quad
& L_i^{\mathrm{base}}
+\sum_{\substack{s\in\mathcal{B}\\i\in\mathcal{C}_s}}
x_{s,i}\Delta L_{s,i}
\leq z^*,
&& \forall i.
\end{aligned}
```

This two-stage objective first minimizes the maximum Engine pressure after the batch is committed, and then picks the allocation with the lower KV recovery cost among the feasible solutions that achieve the same load target.

### Engine-Level Pressure → Path-Level Pressure

In a PD-disaggregated deployment, the actual pressure of a request is jointly determined by the Prefill Engine, the Mooncake KV transfer path, and the Decode Engine. The current scheduler still relies on the local pressure of an individual Engine and therefore cannot represent the end-to-end bottleneck across the P, KV transfer, and D stages in a unified way. A future extension can evolve the existing Engine-level pressure into Path-level pressure: Prefill pressure is computed from request, token, and queue metrics on the Prefill side; Decode pressure is computed from the corresponding metrics on the Decode side; and Mooncake transfer pressure is represented by `kv_transfer_latency_ms`. These three components can then be normalized and combined into a unified pressure for each P→D execution path, allowing the scheduler to select a lower-pressure Prefill and Decode Engine pair without merely shifting the bottleneck to another stage or to the KV transfer path.

## 06 Conclusion

Dressage Proxy Step Balance targets the dynamic long-tail load of agentic rollout and introduces online step-level scheduling on the Proxy side. The scheme avoids synchronous probing on the request path via background load snapshots, corrects snapshot latency with revision-aware unobserved deltas, and constrains greedy selection with context recovery feasibility and a migration gain threshold.

From a systems perspective, this work handles load balancing at a more appropriate layer. The Proxy can simultaneously see session boundaries, token contexts, Engine load, and the rollout request stream, so it can make routing decisions that fit agentic workloads better than static request assignment. Its gain envelope mainly depends on the long-tail degree of the workload, the Engine topology, snapshot freshness, and how faithfully the pressure score characterizes real serving pressure. Therefore, when the existing assignment is already well balanced, the additional performance gain is relatively limited; when hotspots gradually form at runtime, Step Balance can keep correcting with subsequent steps. Within the range where snapshots are valid and candidate Engines meet the migration conditions, the mechanism preferentially steers newly arrived steps to lower-pressure Engines, avoiding pronounced and persistent assignment imbalance as much as possible.
