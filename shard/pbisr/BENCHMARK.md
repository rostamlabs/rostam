# A/B Benchmark: Primary-Backup vs Raft

> **STATUS 2026-09-06 — outcome.** The real-separate-machine gate below, with the
> pipelined engine, found **PB RF=2 beats raft ~1.7× on throughput** (median/p50
> latency also better; note PB's p99 there is comparable-to-slightly-worse, while
> co-located runs show PB p99.9 far better — so the honest tail story is mixed),
> while **PB RF=3 full-ISR loses to raft's majority**. `-replication-mode=pb` is
> promoted out of experimental **primarily on the correctness case** (all three
> blocking hazards closed with passing tests — see DESIGN.md), with **RF=2 as the
> recommended config**; raft stays the default (it wins at RF=3). PERF CAVEAT: the
> gate's raft baseline ran under the OLD epoll inline-dispatch default (see the
> §2026-07-22 epoll section), which the file notes is ~pessimistic for raft, so
> **the RF=2 margin remains unverified under the current goroutine-server default**
> — re-run it on your hardware. The 2026-08 12-vCPU numbers further below are
> PB-vs-Aerospike (no raft control), not a raft reproduction. The "stays
> EXPERIMENTAL / off" conclusions in the historical sections predate the pipelined
> engine and the RF=2 finding — read them as the record of how the verdict
> evolved, not the current status.

This document is the go/no-go **gate** for the dual-mode replication feature
(see `DUAL-MODE-DESIGN.md`). A launchable static 3-node
primary-backup (PB/ISR) cluster alongside the existing per-shard-Raft cluster.
Before any more of the feature is built (quorum-confirmed lease
renewal, automatic failover, ISR shrink/grow), PB must **measurably beat**
Raft on write throughput, on a dedicated box. This is exactly the procedure to
answer that question, plus the launch flags that make it possible.

## Gate criterion

> **PB must measurably beat Raft on write throughput, on a dedicated
> high-clock box, for the failover work to proceed.**
>
> If it does not, STOP: the feature's premise (avoiding per-shard Raft
> consensus overhead pays off) is false, and `-replication-mode=pb` ships
> experimental/off, and the failover work does not proceed.

"Measurably beat" means: PB write throughput clears Raft write throughput by a
margin outside run-to-run noise, at both 128 and 256 connections, on a machine
that is not itself the bottleneck (see "Why not the laptop" below).

## GATE RESULT (2026-07-21, dedicated box) — **FAILED. PB is ~2.3x SLOWER than Raft.**

> **OBSOLETE — superseded 2026-07-22.** This result measured the PRE-pipeline
> engine (one blocking in-flight write per shard). The pipelined engine flips
> it; see "Pipelined engine MEASURED" below before citing anything in this section.

The gate was run on a dedicated Hetzner **CCX53** (32 dedicated vCPU, AMD
EPYC-Milan @ 2.0 GHz, 122 GB) — server processes and load generator sharing the
box but with 32 dedicated cores so nothing is oversubscribed (3 servers ×
`GOMAXPROCS=8` + `netkv` `GOMAXPROCS=6` = 30/32). Both clusters: 3 nodes on
`127.0.0.1`, 8 shards, `replication-factor=3`, `GOGC=40`, HTTP disabled
(`-http ""`). **Durability matched:** Raft ran `-nosync -volatile-log`
(in-memory log, no `write()` on the hot path — the closest match to PB's no-WAL
path); PB ran `-min-isr 2` (commits on the full 3-node ISR). Load: `netkv`
PUT-only, 100k keys, 256-byte values, **30s** measured window, 5s warmup,
**0 errors on every point** (every PB write committed across all 3 ISR members —
so this is genuine replicated throughput, not a single-node path; the
`netkv` engine label "single-node direct" is a stale hardcoded description
string, the engine actually routes to shard leaders/primaries via the topology).

| mode | conns | ops/s | p50 | p99 | p999 |
|---|--:|--:|--:|--:|--:|
| raft (`-nosync -volatile-log`) | 128 | **103,010** | 1.02 ms | 3.09 ms | 20.9 ms |
| raft (`-nosync -volatile-log`) | 256 | **104,063** | 2.05 ms | 5.98 ms | 65.8 ms |
| pb (`-min-isr 2`, full ISR)    | 128 | **43,920**  | 2.62 ms | 7.87 ms | 10.2 ms |
| pb (`-min-isr 2`, full ISR)    | 256 | **44,690**  | 5.23 ms | 13.8 ms | 16.8 ms |

**Verdict: the gate FAILS.** Raft beats PB by ~2.3x on throughput and ~2.5x on
p50/p99 latency, consistently at both connection counts. Per the gate criterion
above, this means **STOP: do not build failover** (quorum-confirmed lease renewal,
automatic failover, ISR shrink/grow). `-replication-mode=pb` stays EXPERIMENTAL
and off by default. The feature's founding premise — that skipping Raft's
per-write consensus log makes primary-backup faster — does not hold as designed
and built.

**Why PB loses (two compounding reasons, both real):**

1. **Full-ISR (3-of-3) commit vs Raft majority (2-of-3).** The OH1 no-acked-loss
   guarantee requires PB to wait for *every* ISR member; Raft commits as soon as
   a *majority* acks. PB waits on the slowest of 3, Raft on the slowest of 2 — a
   structural tax on every write. `-min-isr 2` does NOT relax this (it is only an
   ISR-shrink floor; `Propose` still waits for the full current ISR).
2. **The PB `NetTransport` is a first-cut, un-optimized transport.** One in-flight
   request per connection, no request pipelining, no `writev`/message batching
   (all deferred at the time). Raft's transport is mature: `BatchApplyCh`,
   pipelined `AppendEntries`, in-memory log under `-volatile-log`. The
   consensus-log *savings* PB was supposed to bank are more than eaten by its
   naive transport plus the extra ack it waits for.

**Honest caveat — what could change this, and why it's still not worth it now:**
An optimized PB transport (pipelining + `writev` + batching, the same
Aerospike-inspired work already done for the *Raft* fabric transport) would
narrow the gap. But it would have to overcome a 2.3x deficit *and* the
structural full-ISR-vs-majority headwind — and the only way to remove that
headwind is to commit PB on a majority instead of the full ISR, which reopens
the OH1 acked-write-loss hazard that the full-ISR design exists to prevent. On
this evidence, PB is not a throughput win over Rostam's already-optimized
`-nosync -volatile-log` Raft, and the multi-month failover investment is not
justified. The static PB substrate remains, behind
the experimental flag, for anyone who wants to revisit it with an optimized
transport — but the default path stays Raft.

## Profiling addendum (2026-07-21) — the loss is the transport, not the consensus model

A follow-up CPU/block/mutex profile of a PB node under 128-conn load (captured
locally via `ROSTAM_PPROF`, since a profile's hot-path *breakdown* is far less
clock/oversubscription-sensitive than absolute throughput) answers the one open
question above — *is PB slow for a fixable reason or a structural one?* The
answer is **overwhelmingly the fixable one:**

- **CPU: `shardTransport.Replicate` = 60.5% cumulative; raw `Syscall6` = 65%
  flat**, of which only ~3% is `EpollWait` (the client-facing path) — the rest
  is the PB inter-node transport's per-message `write`/`read`. The node is
  **syscall-bound in the primary-backup transport**, doing **one syscall pair
  per replicated message** because the Plan-3a `NetTransport` is a first-cut with
  **no batching, no pipelining, no `writev`** (all explicitly deferred).
- By contrast Raft's transport amortizes syscalls across **batched** entries
  (`BatchApplyCh` + pipelined `AppendEntries`). That batching — not the absence
  of a consensus log — is why Raft is faster here.
- The block profile's large waits are the **meta group's** Raft heartbeat/pipeline
  loops (background idle), not the PB data path; and mutex contention is in the
  shared client-facing `gnet`/`Node.Call` dispatch (present in both modes). So
  neither points at a PB-specific structural stall.

**Refined conclusion.** The 2.3x deficit is dominated by an **un-optimized
transport**, which is bounded, well-understood engineering (the same
`writev`+pooling+pipelining work already done for the *Raft* fabric transport,
`raft/fabric`), **not** by the primary-backup model or even primarily by the
full-ISR-vs-majority ack difference (which is cheap on co-located/loopback).

**Caveat that keeps the gate honest:** both the throughput A/B *and* this profile
are **co-located** (all nodes on one box / loopback), where the full-ISR "wait
for the slowest of 3" costs almost nothing. On **real separate machines** with
real network RTT, that structural full-ISR-vs-majority penalty would reassert
itself and an optimized transport would help *less* proportionally. So:

- **Do NOT jump to failover** — that was never the bottleneck.
- If PB is worth pursuing at all, the correct, **bounded** next step is: port the
  `writev`/pipelining transport optimizations to `pbisr.NetTransport`, then
  **re-gate on genuinely separate machines** (not loopback). That single
  experiment — small relative to failover — decides whether PB is actually
  competitive. Until it's run, the default stays Raft and `-replication-mode=pb`
  stays experimental/off.

## Follow-up: the batched transport was BUILT and RE-MEASURED — still latency-bound co-located

The pipelined/`writev`-batched `pbisr.NetTransport` (ported from `raft/fabric`;
see `raft/fabric/DESIGN.md`) replaced the per-message transport. Re-measured
co-located (laptop,
128 conns, `-nosync -volatile-log` raft baseline):

| transport | ops/s | p50 | note |
|---|--:|--:|---|
| raft baseline | ~95,000 | 1.13 ms | |
| batched-pb, `linger=0` | ~80,000 | 1.43 ms | ≈ the *naive* pb (~81k) — no change |
| batched-pb, `linger=50µs` | **6,562** | **16.4 ms** | forcing coalescing COLLAPSED it ~12x |

Findings that close the co-located question:
1. **Batched-pb ≈ naive-pb co-located.** A fresh CPU profile of the batched node is
   nearly identical to the naive one (`Syscall6` ~64%). Reason: on loopback the cost
   is the **per-byte data copy through the kernel**, which `writev` batching does not
   reduce — it only amortizes the syscall-*entry* cost, which matters on a real NIC,
   not loopback. And with `linger=0` on a load spread across shards/links, each link
   rarely has >1 frame queued, so batching seldom engages (the micro-benchmark's 50:1
   was a forced single-link burst, not representative of cluster traffic).
2. **`linger>0` is poison here.** Setting `pbWriteLinger=50µs` to *force* batching
   collapsed throughput ~12x and drove p50 to 16ms — the co-located replicated-write
   path is **latency-bound**, and trading round-trip latency for fewer syscalls is a
   catastrophic net loss (exactly why `raft/fabric`'s `writeLinger` defaults to 0).
3. **Therefore the residual gap to Raft is structural, not transport.** It is the
   full-ISR (3-of-3) vs Raft-majority (2-of-3) commit, which no transport change
   removes. Batching helps only a **CPU/syscall-bound** deployment (real separate
   machines, saturated cores) AND needs a `linger` value that helps without wrecking
   latency — a narrow window still sitting under the full-ISR headwind.

**Bottom line:** across every configuration measured (naive & batched; `linger` 0 &
50µs; laptop & the dedicated box — all co-located/loopback), **Raft beats PB.** The
batched transport is correct and stays on the branch behind the experimental flag for
a future real-separate-machine investigation, but on this evidence PB is not a
throughput win over Rostam's `-nosync -volatile-log` Raft, and the default stays Raft.

## Pipelined engine MEASURED (2026-07-22, dev laptop) — the pipelined engine FLIPS the verdict; RF=2 is the headline

The pipelined write path (PIPELINE-REDESIGN) was benchmarked
for the first time, plus group frames + cumulative acks. All numbers: dev laptop (i9-13900H,
P/E hybrid, 20 logical cores, co-located 3-node + netkv, the usual
directional-only caveats), 8 shards, `netkv` PUT-only, 100k keys, 256 B
values, 30 s window, raft baseline `-nosync -volatile-log`, **0 errors on
every point**.

**Fresh machine state (morning, idle box):**

| mode | conns | ops/s | p50 | p99 | p999 |
|---|--:|--:|--:|--:|--:|
| raft | 128 | 102,101 | 0.96 ms | 3.9 ms | 40.4 ms |
| pb pipelined RF=3 (full-ISR 3-of-3) | 128 | **143,776** | 0.76 ms | 2.1 ms | 3.4 ms |
| raft | 256 | 104,582 | 2.01 ms | 6.8 ms | 45.2 ms |
| pb pipelined RF=3 | 256 | **142,573** | 1.61 ms | 4.0 ms | 6.3 ms |

PB beats raft ~1.4x on throughput with better latency at every percentile
(p999 ~10x better — raft's tail carries its heartbeat/election machinery).
The 2026-07-21 "PB is 2.3x slower" gate result was measured on the
PRE-pipeline engine and is obsolete: the deficit was round-trip
serialization (one in-flight write per shard), not the PB model.

**Degraded machine state (afternoon: thermal + memory pressure; raft
controls at 62-66k confirm uniform conditions within each battery):**

| mode | conns | ops/s | p50 | note |
|---|--:|--:|--:|---|
| raft control (start / end) | 128 | 62,458 / 66,230 | ~1.6 ms | stable controls |
| pb RF=3 pre-Plan-G, 2 runs | 128 | 53,553 / 53,072 | ~2.1 ms | interleaved |
| pb RF=3 with group frames, 2 runs | 128 | 55,619 / 53,732 | ~2.1 ms | interleaved |
| **pb RF=2 with group frames** | 128 | **175,595** | **0.59 ms** | p999 4.0 ms; 2nd run 171,464 |

Findings, stated honestly:

1. **RF=2 (the Aerospike-parity config: primary + 1 backup, wait on the one
   backup's in-RAM ack) is the headline: ~2.7x raft throughput at ~1/3 the
   p50, stable across machine states** (171k/175k on the degraded box —
   faster than anything else measured on this laptop, either mode, either
   state). Research note: Aerospike's default `commitLevel=all` at RF=2 is
   exactly this contract (wait for every replica's in-memory ack; fsync is
   opt-in and off the ack path), so this is a like-for-like posture.
2. **PB RF=3 is bimodal on this laptop** where raft is not: it beats raft
   ~1.4x on a fresh box and sits ~15% under raft when thermally/memory
   degraded (both pre- and post-Plan-G — waiting on the slowest of TWO
   oversubscribed backups amplifies scheduling jitter; raft's majority
   commit waits on the faster of its two followers). PB's p999 stays far
   better than raft's in every state.
3. **Group frames are durability-neutral and performance-neutral co-located**
   (interleaved A/B above: parity within noise). This matches the earlier
   linger-sweep finding: loopback is per-byte-copy-bound, so fewer
   frames/acks/locks per write don't move co-located throughput. Its win is
   architectural (per-message costs amortized k-fold) and must be proven on
   real separate machines with a real NIC — the same caveat as every other
   number in this file. Opus review verdict: durability-correct, no path
   falsely commits; one liveness hardening (byte-capped groups) applied.
4. The decisive gate remains a dedicated-box, real-network re-run per
   "Running the decisive gate run" below — now with THREE configs (raft,
   pb RF=3, pb RF=2) and the pipelined+Plan-G engine.

## DECISIVE GATE (2026-07-22, real separate machines) — PB RF=2 WINS 1.7x; PB RF=3 full-ISR loses

The real-network gate the 2026-07-21 verdict called for was run: 4x Hetzner
CCX33 (8 dedicated vCPU each) in fsn1 — three server nodes (one rostam
process each, GOMAXPROCS=8, GOGC=40, 8 shards) plus a SEPARATE load box, all
traffic over the private network. Engine: pipelined + group frames.
netkv PUT-only, 100k keys, 256 B
values, 30 s window, 5 s warmup, 3 repeats per point (median reported;
run-to-run spread was ~2%, dramatically tighter than any laptop number in
this file). Raft baseline `-nosync -volatile-log`. **0 errors on every run.**

| config | conns | ops/s (median of 3) | p50 | p99 |
|---|--:|--:|--:|--:|
| **pb RF=2** (`-replication-mode pb`, 2 owners) | 128 | **61,109** | **1.35 ms** | 8.2 ms |
| raft RF=3 | 128 | 35,386 | 3.30 ms | 7.4 ms |
| pb RF=3 (full-ISR 3-of-3) | 128 | 20,927 | 5.62 ms | 12.0 ms |
| **pb RF=2** | 256 | **62,103** | **2.31 ms** | 16.4 ms |
| raft RF=3 | 256 | 35,937 | 6.71 ms | 14.5 ms |
| pb RF=3 (full-ISR 3-of-3) | 256 | 20,652 | 11.75 ms | 24.2 ms |

**Verdict, two halves:**

1. **PB RF=2 — the Aerospike-parity contract (client → primary → ONE
   backup's in-RAM ack) — PASSES the gate decisively: ~1.7x raft throughput
   at ~40% of raft's p50, at both connection counts, reproducibly.** This is
   the configuration that competes with Aerospike (whose default
   `commitLevel=all` at RF=2 is the same wait-for-the-one-replica posture),
   and it is the fastest replicated configuration this codebase has ever
   measured on fair hardware.
2. **PB RF=3 full-ISR FAILS against raft RF=3 over a real network (~0.58x),
   confirming the original caveat:** waiting on the SLOWEST of two backups
   costs more than raft's majority commit (fastest of two followers), and no
   transport work changes that structure. Anyone wanting PB at RF=3
   competitively needs a quorum-commit option — a durability-contract
   change, deliberately out of scope for group frames.

Same-durability note: all three configs above commit on in-memory
replication acks only (raft runs `-nosync -volatile-log`; PB has no WAL).
RF=2 PB holds one fewer copy than the RF=3 configs — that is a capacity and
fault-tolerance tradeoff (survives 1 node loss, like raft RF=3's quorum
availability, but with 2 copies not 3), priced into its win.

Cost of the gate: 4x CCX33 for ~1 h ≈ EUR 1.10.

## Aerospike head-to-head (2026-07-22, same topology) — Aerospike leads 2.6-3.0x on replicated writes

Same 4x CCX33 layout (3 server nodes + separate load box, private net), same
netkv load (PUT-only, 100k keys, 256 B, 30 s, 3 repeats): **Aerospike CE
8.1.2.3**, 3-node cluster (docker `--network host`, in-memory namespace,
`replication-factor 2`), driven with `netkv -sync` = **`COMMIT_ALL`** — the
same wait-for-the-replica's-in-RAM-ack contract as Rostam PB RF=2. 0 errors.

| engine (both RF=2, sync ack) | conns | ops/s (median) | p50 | p99 | p999 |
|---|--:|--:|--:|--:|--:|
| Aerospike CE | 128 | **157,186** | 0.73 ms | 2.07 ms | 3.3 ms |
| Rostam PB (8 shards) | 128 | 61,109 | 1.35 ms | 8.2 ms | 10.5 ms |
| Aerospike CE | 256 | **188,701** | 1.17 ms | 5.0 ms | 9.0 ms |
| Rostam PB (8 shards) | 256 | 62,103 | 2.31 ms | 16.4 ms | 21.3 ms |

**Gap: 2.6x at 128 conns, 3.0x at 256; ~1.8-2x on p50.** Aerospike also
SCALES with offered load (157k -> 189k from 128 -> 256 conns) where Rostam is
flat (61k -> 62k) — a saturation ceiling on our side.

**The ceiling is NOT shard serialization:** a 32-shard Rostam PB RF=2 run on
the same cluster moved the needle only ~6% (64.3-65.8k, still flat 128->256).
And it is NOT the server/protocol stack per se: single-node, same harness,
Rostam BEATS Aerospike ~1.3x on PUTs (netkv README). The deficit is
specifically the replicated write path's per-write latency — the mean
per-op time under load is ~2 ms vs Aerospike's ~0.8 ms, consistent with the
goroutine-handoff chain each write crosses (gnet loop -> shard dispatch ->
Propose -> per-peer sender -> link writer -> [wire] -> backup reader ->
backup writer -> [wire] -> link reader -> completion sweep -> doneCh wakeup:
~6 scheduler wakeups per write). Candidate levers, in likely-impact order,
all unproven until profiled ON THIS PATH under real-network load:

1. pprof the primary under the gate load (CPU + block + sched traces) —
   confirm where the ~1.2 ms goes before building anything.
2. Cut wakeups: submit the replicate frame inline from the Propose goroutine
   (skip the per-peer sender hop when the link send queue is uncontended);
   complete Propose via the ack callback instead of a per-write channel.
3. Client-conn pipelining (>1 op in flight per conn) if the client protocol
   allows — Aerospike's client is also 1-op-per-conn here, so this is a
   fairness-neutral lever only if theirs is doing the same.

Verdict: the PB/ISR *model* is right (it beat Raft decisively) and the
single-node engine is faster than Aerospike's; closing the remaining
2.6-3.0x replicated-write gap is per-write-latency engineering on the
pbisr data path, with profiling as the mandatory first step.

## FOUND IT: the epoll inline-dispatch ceiling (2026-07-22, same day) — one flag closes most of the gap

The mandated profile (CPU + block + goroutine dump under 128-conn PB RF=2
load, `ROSTAM_PPROF`) found the ceiling immediately: **the goroutine dump
showed ~18 goroutines parked in `Engine.Propose`'s commit wait — ON THE GNET
EVENT-LOOP GOROUTINES.** `EpollServer.OnTraffic` executes `dispatch`
(→ `Store.Call` → `Propose`) INLINE on the event loop, so every replicated
write parks an entire loop for a full replication round trip and every other
connection mapped to that loop stalls behind it. Loopback hides it (RTT
~40µs); a real network (~300µs RTT) is catastrophic: max write throughput
becomes ~(loops / RTT) — independent of client connections and shard count —
which is exactly the measured flat 61-65k.

`-epoll=false` (the goroutine-per-connection server: a blocked write parks
only its own connection) on the SAME 4x CCX33 real-network topology,
PB RF=2, 8 shards, medians of 3 (~1% spread, 0 errors):

| engine (RF=2, sync replica ack) | conns | ops/s | p50 | p99 | p999 |
|---|--:|--:|--:|--:|--:|
| Aerospike CE | 128 | **157,186** | **0.73 ms** | 2.07 ms | 3.3 ms |
| Rostam PB, goroutine server | 128 | 127,235 | 0.96 ms | **1.97 ms** | **2.9 ms** |
| Rostam PB, epoll (old default) | 128 | 61,109 | 1.35 ms | 8.2 ms | 10.5 ms |
| Aerospike CE | 256 | **188,701** | **1.17 ms** | 5.0 ms | 9.0 ms |
| Rostam PB, goroutine server | 256 | 145,196 | 1.68 ms | **3.2 ms** | **4.8 ms** |
| Rostam PB, epoll (old default) | 256 | 62,103 | 2.31 ms | 16.4 ms | 21.3 ms |

(One 512-conn point: 155k ops/s — still scaling with offered load.)

**Where this lands vs Aerospike: throughput gap cut from 2.6-3.0x to
1.24x/1.30x, and Rostam's write TAIL is now BETTER than Aerospike's at both
connection counts (p99 1.97 vs 2.07 ms @128; 3.2 vs 5.0 ms @256).** Both
engines are latency-bound at fixed connection count (ops/s ≈ conns/p50), so
the remaining deficit is ~230µs of per-op latency (0.96 vs 0.73 ms p50
@128). The identified next levers, in order: (1) inline the replicate submit
from the Propose goroutine (skip the per-peer sender hop when the link queue
is uncontended); (2) complete Propose via the ack callback instead of the
per-write doneCh channel wakeup; (3) revisit an epoll transport that hands
off blocking ops to workers and replies via AsyncWrite (best of both).

**Lever results (2026-07-22, co-located interleaved A/Bs, 128 conns, PB RF=2):**

- **Lever 1 — inline replicate submit (COMMITTED):** the Propose
  goroutine submits the frame to the peer link itself (skipping the per-peer
  sender wakeup) when the link is connected and NOTHING for that peer is
  queued or in flight on the sender path (a per-peer pending count is the P1
  ordering gate). ~+8% ops/s (307/281k -> 328/311k), better p99/p999.
- **Lever 2 — inline WIRE writes (TRIED, REVERTED):** writing the frame (and
  the backup's ack) directly on the submitting goroutine via a try-locked
  conn write, skipping the framed-writer goroutine, measured **-18%**
  (312/324k -> 263/255k, worse at every percentile). Mechanism: the write
  syscall lands inside the per-shard writeMu sequencing section on the
  primary and serializes the shared peer-link reader on the backup —
  trading away the writer goroutines' pipelining under load for one saved
  wakeup. Not pursued; any future attempt needs a load-aware gate.
- **Fix #2 — backup inline-ack, the reorder-safe half of lever 2 (TRIED,
  REVERTED, 2026-07-23):** re-attempt of ONLY lever 2's backup side, on the
  premise its −18% came from the primary-side writeMu serialization (absent
  on the backup) plus per-frame bookkeeping — so a load-gated
  (`len(sendCh)==0`) try-locked inline ack, no counter, should be neutral
  co-located and win on the wire by dropping a park/unpark. Measured
  **−16%** co-located (299k→251k, p50 352→419µs) — the SAME signature as
  lever 2. Mechanism confirmed for the backup independently: the dedicated
  writer goroutine (a) coalesces acks into fewer `writev`s and (b) runs the
  send syscall on a spare core while the reader goroutine immediately reads
  the next replicate request; inlining defeats both and serializes
  apply-then-send on one goroutine. The backup has spare cores on the wire
  too, so this predicts a wire regression as well — the wire A/B was NOT run
  (co-located signal decisive, unlike lever 1 where loopback hid the win).
  VERDICT: the goroutine-per-conn writer with ack batching is the right
  design on the backup; do not inline acks. Reverted; tree unchanged.
- **GOGC sweep:** 150 ≈ 40 (GC is not the current bottleneck); `GOGC=off`
  collapses (memory growth -> errors). Keep 40.
- **Lever 1 WIRE verification (4x CCX33, medians of 3):** 130.4k @128
  (p50 0.95 ms, p99 **1.77 ms**) / 141.5k @256 — throughput WITHIN NOISE of
  pre-lever (127.2k/145.2k), p99 improved ~10%. The saved wakeup (~10µs) is
  invisible against a ~950µs wire p50; lever 1's throughput win is co-located
  only. Standing gap to Aerospike: **1.21x @128, 1.33x @256** on throughput
  (Rostam's p99/p999 remain better). The remaining credible lever is the
  async epoll server (worker handoff + AsyncWrite — "lever 3"), which
  attacks the per-connection blocking model itself; smaller candidates:
  read-path buffer pooling, client-side connection scaling.

**Alloc-rate + contention rounds (2026-07-22 evening, all local per
constraint; interleaved A/Bs at 128 conns, PB RF=2, loopback):**

- **Alloc cuts round 1 (COMMITTED):** pending map -> inline array,
  decode alias, ProposeDeadline w/ pooled timer. Alloc rate 3.4 -> 1.7GB/15s
  (-49%), +2.7% ops.
- **Alloc cuts round 2 (COMMITTED):** log-entry buffers pooled via
  the ring-eviction release hook (WithDataRelease); serveConn read buffers
  pooled (recycled after the receiver returns — backed by a full write-
  handler retention audit: every handler copies at decode or page-write).
  +2.5% ops, better p99. Cumulative local: ~294k -> ~335k ops/s (+14%).
- **Control-snapshot move (COMMITTED with this note):** the 3 MetaFSM reads
  (global lock, ISR copy alloc) moved OUT of the writeMu serialized section;
  isolated effect within noise, kept on mechanism (mutex profile showed the
  writeMu chain at ~4% of runtime; the reads gave it cross-shard coupling).
- **Async epoll server ("lever 3") — TRIED, REVERTED:** classify blocking ops
  on the loop, dispatch them on their own goroutine, reply via AsyncWrite.
  Measured -60% PUT-only (123-137k vs 334-340k) and -38% at 50/50 read-write
  (234k vs 376k): every AsyncWrite re-enters the event loop (a wakeup +
  cross-loop queue per response) plus a goroutine spawn per request —
  strictly worse than goroutine-per-connection's direct write. VERDICT: the
  goroutine server IS the cluster transport; the epoll path stays for
  single-node/read-heavy use only, and no async variant is worth building.
- **Inflight record/doneCh pooling — DELIBERATELY SKIPPED:** a record can be
  resolved-but-unpopped (failed record awaiting head-pop) while its waiter
  has already returned, so recycling needs a two-owner refcount on the
  correctness-critical commit path; ~350MB/15s of the remaining alloc rate
  does not justify that risk.

**Local co-located head-to-head vs Aerospike (2026-07-22 late — the free,
directional comparison after the do-all sweep):** 3-node Aerospike CE 8.1
(docker host-net, in-memory, RF=2, `COMMIT_ALL`, per-node access-port
advertisement) vs 3-node Rostam PB RF=2, both co-located with netkv on the
dev laptop, interleaved twice, 128 conns, PUT-only, 0 errors:

| engine (RF=2, sync ack) | ops/s (2 runs) | p50 | p99 | p999 |
|---|--:|--:|--:|--:|
| **Rostam PB** | **309.2k / 320.9k** | 347-353 µs | **1.2-1.3 ms** | **1.9-2.3 ms** |
| Aerospike CE | 188.8k / 171.8k | 386-432 µs | 4.9 ms | 9.8 ms |

**Rostam leads 1.7-1.8x co-located, at better p50 and ~4x better p99.**
Read together with the real-network gate, the two regimes now tell one
consistent story: Aerospike's LOCAL numbers (172-189k) match its WIRE
numbers (157-189k) — it is pipeline/RTT-bound and insensitive to
co-location — while Rostam's per-op software cost is lower (loopback
strips the wire and we win big) but our wire pipeline still pays more per
RTT (they lead 1.21-1.30x there). The remaining work to beat Aerospike on
a real network is therefore wire-path work (per-RTT cost), not engine CPU
— consistent with the lever-1 wire result.

**Final wire gate after the do-all sweep (2026-07-23, 4x CCX33, medians of
3, ~1% spread, 0 errors):** PB RF=2 with the full engine (group frames + lever 1 +
both alloc rounds + ctrl-move): **131.6k @128 (p50 0.95 ms, p99 1.63 ms),
140.1k @256 (p99 3.1 ms), 155.8k @512** — throughput IDENTICAL to the
pre-alloc-round wire run (130.4k/141.5k), p99 a further ~8% better (1.63 vs
1.77 ms). This CONFIRMS the two-regime model: the engine-CPU work moved
loopback +14% and wire tails, but wire THROUGHPUT is pinned by per-RTT
pipeline cost. Standing vs Aerospike (157.2k/188.7k): **0.84x @128,
0.74x @256 — with Rostam ahead on p99 at both (1.63 vs 2.07 ms;
3.1 vs 5.0 ms)**. Engine-side levers are exhausted for this gap; what
remains is wire-pipeline work: client-side pipelining/multiplexing (>1 op
in flight per conn — a client+protocol change), and response-path syscall
reduction on the client-facing conn. Nothing else in the server moves it.

**Pipelining gate (2026-07-23, 4x CCX33) — the conn model is exonerated; a
~157k server-side wall found with the primary 76% IDLE:** the server now
supports per-connection request pipelining with ordered responses
(strict request-response clients keep the byte-identical inline
path), and netkv gained a pipelined shard-routing client (rostam-bench). On
the wire, 512 workers over just 12 pipelined conns match 512 classic conns
EXACTLY at every load: 129.9k/144.9k/156.5k/157.0k at 128/256/512/1024
workers, latency growing linearly while throughput stays flat — classic
queueing at a fixed-capacity stage. 16 shards: identical (156.8k). The
saturation profile shows the PRIMARY at 24% CPU (syscall + futex +
scheduler churn, no hot work). So the wall is NOT: client conns, client
model, shard count, primary CPU, or (per the earlier rounds) allocations.
Remaining suspects, in probe order for the next session: (1) the load
generator's own per-op CPU (both rostam client paths share the
EncodePutArgs/EncodeRequest stack; the leaner Aerospike client reached
188.7k on the same box), (2) the single inter-node PB link + the backup's
serial serveConn receive path. Until the generator is exonerated, the
honest statement is "Rostam sustains >=157k RF=2 writes/s on 3x8 vCPU with
its primary 3/4 idle" — the true server ceiling is not yet known.

## TWO-GENERATOR PROBE (2026-07-23) — every prior wire number was GENERATOR-limited; true ceilings are near-parity

The probe that names the real ceilings: 3 server nodes + TWO separate load
boxes (5x CCX33), both engines RF=2 sync-replica-ack, 512 workers per
generator, PUT-only, 0 errors:

| engine | 1 gen (control) | 2 gens simultaneously (sum, 2 reps) | p99 at saturation |
|---|--:|--:|--:|
| Rostam PB (classic client) | 153.3k | **284.6k / 282.1k** | **7.5-8.3 ms** |
| Aerospike CE | 181.4k | **314.6k / 288.0k** | 12.4-12.5 ms |

1. **EVERY single-generator wire number in this file was a
   load-generator ceiling, for BOTH engines** (Rostam ~157k -> ~284k;
   Aerospike ~189k -> ~288-315k). One 8-vCPU netkv box cannot saturate
   either 3-node cluster.
2. **True saturation on 3x8vCPU RF=2: near-parity.** Aerospike sums
   288-315k (noisy, ±9%), Rostam 282-285k (stable, ±1%): a 1-10% Aerospike
   edge on peak throughput — down from 2.6-3.0x this morning — with
   **Rostam's saturation p99 ~1.5x better (8.0 vs 12.5 ms)**.
3. The earlier "~157k wall with the primary 76% idle" is fully explained:
   the wall was the generator; the 76%-idle primary was the tell. The
   pipelining round's conclusion stands: the conn model was never the
   limit, and Rostam's server had headroom all along.
4. Method note for all future gates: replicated-write comparisons at this
   scale need >= 2 generator boxes, and the per-engine client's own CPU
   cost shifts single-generator results (the leaner Aerospike client got
   further on one box than ours).

## CPU-MEASURED PROBE (2026-07-23) — BOTH servers are ~95% IDLE at 300k; the generators were ALWAYS the wall

The probe that ends the guessing: per-node `mpstat` idle% captured DURING
dual-generator saturation, both engines RF=2 sync-replica-ack, 512
workers/generator.

| run | throughput | primary idle% | 2nd node idle% | p99 |
|---|--:|--:|--:|--:|
| Rostam PB, classic client | 304.5k (152+153) | **97.5%** | 92.4% | 7.8 ms |
| Rostam PB, pipelined client | 298.8k | **97.6%** | 92.2% | 8.3 ms |
| Aerospike CE (2-node) | 199.6k | **98.3%** | 94.8% | 17-22 ms |

Findings — this supersedes every throughput ranking earlier in this file:

1. **The server was NEVER the bottleneck, for either engine.** Rostam's 3
   nodes serve 300k replicated writes/s using ~5% of 24 vCPU; a fresh CPU
   profile of the primary at that load is ~2.4% busy, entirely
   epoll/scheduler, ZERO application hot path. Aerospike is likewise ~95%
   idle. Every single-cluster number in this file (Rostam's "157k wall",
   Aerospike's "189k") was a LOAD-GENERATOR ceiling: one 8-vCPU netkv box
   caps ~150k ops/s (its own encode+TCP+goroutine CPU), independent of
   pipelining or shard count.
2. **Neither server's true throughput ceiling has been reached** on this
   account (5-server quota → max 2 generator boxes; both servers stayed
   idle even so). The honest statement is: **Rostam sustains >=300k RF=2
   writes/s on 3x8 vCPU with every node >90% idle** — its real ceiling is
   far higher and unmeasured.
3. **Where the engines CAN be compared generator-independently — tail
   latency and server efficiency — Rostam leads:** at saturation Rostam's
   p99 is ~8 ms vs Aerospike's 17-22 ms (~2.5x better), and Rostam's per-op
   server CPU is negligible. The "Aerospike is 1-10% faster" reading from
   the two-generator round was within generator noise, not a server
   property.

**Bottom line: on identical hardware, both are so server-efficient that the
benchmark harness is the limit, not the datastore. Rostam is not slower than
Aerospike in any measurement that isolates the server; it is decisively
better on tail latency. Proving an absolute throughput winner needs ~8-10
generator boxes (beyond this account's quota) — a pure harness-scaling
exercise, not an engine question.**

**Default changed (cmd/rostam-server/main.go):** in `-cluster` mode the
effective default is now the goroutine server; an explicit `-epoll` still
forces the event-loop transport. Single-node default unchanged (epoll on —
its low-concurrency advantage there is real and reads never block). NOTE:
the raft-mode gate numbers in this file were also measured under the old
epoll default and are therefore ~pessimistic; any future raft/PB A/B should
re-run both modes on the new default.

## Launch flags (this task's deliverable)

`cmd/rostam-server/main.go` now wires:

- `-replication-mode raft|pb` (default `raft`) → `cluster.Config.ReplicationMode`.
  Default is `raft` (or unset), which is **byte-identical** to the pre-existing
  path — only `pb` changes behavior.
- `-min-isr N` → `cluster.Config.MinISR`. Required (`>= 1`) when
  `-replication-mode=pb`; ignored in `raft` mode.
- `-pb-addr host:port` → this node's own `Peer.PBAddr` (the node's
  `pbisr.NetTransport` listen endpoint). Required in `pb` mode.
- `-peers` accepts an optional 4th `@`-separated field for a peer's `PBAddr`:
  `id@raftAddr@serverAddr@pbAddr`. The 3-field form still works everywhere
  (`PBAddr` empty) — needed for `raft` mode and for any peer that isn't part of
  a `pb` cluster.

Misconfiguration fails loudly at startup: `cluster.Config.Validate()` rejects
an unknown `-replication-mode` value and requires `-min-isr >= 1` in `pb`
mode; `cluster.newMultiNode` errors if this node's `PBAddr` is empty in `pb`
mode.

## Procedure: 3-node cluster, each mode

Identical topology, shard count, `GOGC`, and load for both runs — only
`-replication-mode` (and the PB-only flags it requires) differs.

**Durability parity is mandatory, not optional.** The PB write path
(`shard/pb_applier.go`'s `shardApplier.Apply` → `fsm.applyEntryData`) is a pure
in-memory FSM apply: there is no per-shard WAL and no fsync anywhere on the PB
hot path (see `shard/pbisr/engine.go`'s `Propose` — it commits on in-memory ISR
network acks, never a disk write). If the Raft baseline runs at this repo's
*default* durability (fsync per write, `-nosync` unset), the A/B stops being a
replication-mode comparison and becomes a "disk-latency-bound Raft vs
pure-memory PB" comparison — the fsync confound will dominate any real
structural difference between the two replication designs. **The Raft baseline
MUST be launched with `-nosync`** so both sides commit on replication alone
with no local disk write in the critical path; that is the only way this A/B
is fair, because the PB path is inherently nosync and cannot be made to fsync.
For an even closer match on the data shards, add `-volatile-log` (in-memory
Raft log, no `write()` syscall at all on the hot path — see the flag's own doc
comment for the fresh-rejoin safety caveat this requires operationally); at
minimum `-nosync` is required, `-volatile-log` is the closer match where you
can afford its rejoin constraint.

### 1. Raft mode (baseline)

```sh
PEERS="n1@10.0.0.1:7400@10.0.0.1:7000,n2@10.0.0.2:7400@10.0.0.2:7000,n3@10.0.0.3:7400@10.0.0.3:7000"

GOGC=40 rostam-server -cluster -bootstrap -node-id n1 -raft-addr 10.0.0.1:7400 \
  -tcp 10.0.0.1:7000 -shards 8 -replication-factor 3 -nosync \
  -data /var/lib/rostam-bench/raft-n1 -peers "$PEERS"

GOGC=40 rostam-server -cluster -node-id n2 -raft-addr 10.0.0.2:7400 \
  -tcp 10.0.0.2:7000 -shards 8 -replication-factor 3 -nosync \
  -data /var/lib/rostam-bench/raft-n2 -peers "$PEERS"

GOGC=40 rostam-server -cluster -node-id n3 -raft-addr 10.0.0.3:7400 \
  -tcp 10.0.0.3:7000 -shards 8 -replication-factor 3 -nosync \
  -data /var/lib/rostam-bench/raft-n3 -peers "$PEERS"
```

(Add `-volatile-log` alongside `-nosync` on all three nodes for the closer
data-shard match, if you can accept its fresh-rejoin-only-on-crash
constraint.)

### 2. PB mode (candidate)

Same shard count, same `GOGC`, same replication factor — add
`-replication-mode pb -min-isr 2`, give each node a `-pb-addr`, and extend the
`-peers` spec with the 4th `@pbAddr` field so every node can resolve every
other node's PB transport address:

```sh
PEERS="n1@10.0.0.1:7400@10.0.0.1:7000@10.0.0.1:7200,n2@10.0.0.2:7400@10.0.0.2:7000@10.0.0.2:7200,n3@10.0.0.3:7400@10.0.0.3:7000@10.0.0.3:7200"

GOGC=40 rostam-server -cluster -bootstrap -node-id n1 -raft-addr 10.0.0.1:7400 \
  -tcp 10.0.0.1:7000 -shards 8 -replication-factor 3 \
  -replication-mode pb -min-isr 2 -pb-addr 10.0.0.1:7200 \
  -data /var/lib/rostam-bench/pb-n1 -peers "$PEERS"

GOGC=40 rostam-server -cluster -node-id n2 -raft-addr 10.0.0.2:7400 \
  -tcp 10.0.0.2:7000 -shards 8 -replication-factor 3 \
  -replication-mode pb -min-isr 2 -pb-addr 10.0.0.2:7200 \
  -data /var/lib/rostam-bench/pb-n2 -peers "$PEERS"

GOGC=40 rostam-server -cluster -node-id n3 -raft-addr 10.0.0.3:7400 \
  -tcp 10.0.0.3:7000 -shards 8 -replication-factor 3 \
  -replication-mode pb -min-isr 2 -pb-addr 10.0.0.3:7200 \
  -data /var/lib/rostam-bench/pb-n3 -peers "$PEERS"
```

`-min-isr 2` is **not** what determines how many replicas each write waits
for here, and it does **not** make this an equivalent durability contract to
Raft. `pbisr.Engine.Propose` (`shard/pbisr/engine.go`) always waits for every
member of the shard's *current* ISR, and the bootstrap seed
(`bootstrapPBShardControl`) sets each shard's ISR to **all** of its owners —
with RF=3 that's all 3 nodes. So in this static cluster every PB write waits
for 3-of-3 in-memory acks, not 2-of-3: `-min-isr 2` is only a floor below
which the control plane would refuse to shrink the ISR further (an ISR-shrink
guard for the then-unbuilt failover path); it is inert here and
plays no role in how many acks a write waits for.

More importantly, "no acked write is lost if one node dies" is
**replication-durability**, not **local crash-durability**, and those are not
the same guarantee. Default-mode Raft (fsync per write) additionally survives
a *power loss* of the very node that took the write — the entry is on disk
before it acks. Nosync PB (like nosync Raft) does not: an acked write lives
only in the acking processes' memory until the next snapshot/compaction, and
a simultaneous loss of every acking node loses it. PB does not have a
disk-durable mode to fall back to (there is no WAL on the PB path at all), so
the only way to run a fair A/B is to put Raft in the *same* posture —
`-nosync` — and compare replication-durability against replication-durability
on both sides, as required above.

Give the PB cluster a few seconds after bringup before driving load: the
control-plane seed (`bootstrapPBShardControl`) and the static lease-keeper
need one refresh cycle to grant each shard's primary its lease (see
`cluster/pb_wiring.go`'s `leaseKeeper` doc comment — this is expected startup
behavior, not a bug).

### 3. Load: identical `netkv` run against each cluster

Point the `netkv` bench (`rostam-bench/netkv/`) at any node's
`-tcp` address — the smart client discovers shard leaders/primaries and routes
writes there — and run PUT-only at both 128 and 256 connections against each
cluster in turn:

```sh
go build -o netkv ./netkv
GOMAXPROCS=<leave headroom for the servers> ./netkv -engine rostam -rostam <node1-tcp> \
  -conns 128 -duration 30 -warmup 5 -readpct 0 -keys 100000 -valsize 256
GOMAXPROCS=<leave headroom for the servers> ./netkv -engine rostam -rostam <node1-tcp> \
  -conns 256 -duration 30 -warmup 5 -readpct 0 -keys 100000 -valsize 256
```

Record ops/sec and p50/p99 for each `(mode, conns)` pair. That is the whole
comparison — same generator, same keyspace, same durability contract, only
`-replication-mode` differs between the two cluster bringups.

## Why the laptop cannot produce the decisive number

Per `rostam-benchmark-hardware`: this development machine is a P/E-core
hybrid laptop that exhibits a 1.27–1.82x throughput swing at low concurrency
from core-scheduling lottery alone, plus thousands of thermal-throttle events
under sustained load, and it is shared between the 3 server processes *and*
the load generator (co-location inflates/deflates whichever side the
scheduler favors that run). Any single-machine number here is **directional
only**, and even the *shape* is only informative once both modes are run at
matched durability (`-nosync` on the raft baseline, per above) — until that
variable is controlled, an observed gap cannot be attributed to Raft's
consensus/heartbeat/election tax at all, because an un-fsync'd disk-durability
difference is sitting on top of it and dominates. The structural argument for
PB's advantage is modest, not dramatic: primary-backup replication skips
Raft's per-write log-matching and commit-notification round trips relative to
a single primary→ISR replicate-and-ack hop, which predicts something in the
neighborhood of **2-2.4x**, not 20x or 53x. A fsync-matched run is expected to
land far closer to that structural range than to this document's uncontrolled
numbers below. The gate must be run on a
dedicated, high-clock box with the server processes alone on their cores and
the load generator elsewhere (or on cores the servers do not use), exactly as
the existing `netkv` fairness controls already require for every other engine
comparison in that harness.

## Directional numbers captured on this laptop — INVALID, kept for provenance only

> **These numbers do NOT answer the gate question and must not be cited as
> evidence of PB's magnitude of advantage over Raft.** They are kept below
> only so the next run has a documented starting point and so nobody
> re-derives the same mistake. See "The flaw in this run" immediately below
> before reading the table.

Captured 2026-07-21 on the dev laptop (13th Gen Intel i9-13900H, 20 logical
cores, P/E hybrid — see caveat above). Both clusters: 3 nodes, all on
`127.0.0.1`, 8 shards, `replication-factor=3`, `GOGC=40`, `GOMAXPROCS=6` per
server process (leaving headroom for the co-located `netkv` client, itself
capped at `GOMAXPROCS=4`), PB run with `-min-isr 2`. Load: `netkv -readpct 0`
(PUT-only), 50,000 keys, 256-byte values, 5s measured window (short window —
a real gate run should use ≥30s). **0 errors on every point.**

| mode | conns | ops/s | p50 | p99 | p999 |
|---|--:|--:|--:|--:|--:|
| raft | 128 | 4,666 | 24.0 ms | 59.1 ms | 70.9 ms |
| raft | 256 | 1,740 | 56.5 ms | 728.5 ms | 850.5 ms |
| pb   | 128 | 93,960 | 1.21 ms | 3.56 ms | 4.85 ms |
| pb   | 256 | 92,900 | 2.54 ms | 6.35 ms | 8.11 ms |

**The flaw in this run:** the raft row above was captured at this repo's
*default* durability — fsync per write, **`-nosync` was not passed** — while
the pb row has no WAL/fsync on its write path at all (see the note above the
launch commands). This is not a replication-mode A/B, it is a
"disk-latency-bound Raft vs pure-in-memory PB" A/B, and the fsync gap
dominates the result: this same codebase's other Raft measurements at
`-nosync` (in-memory log, e.g. `rostam-epoll-transport`'s ~140k ops/s
figures) sit roughly **30x above the 4,666 ops/s raft row here**, which is
the size of tax a per-write fsync to disk is expected to cost, not something
attributable to Raft's consensus/heartbeat machinery. The 256-conn raft
collapse (4,666 → 1,740 ops/s, more than halved) is a second, separate
confound: 3 server processes plus the load generator sharing 20 contended
P/E cores on this laptop, not a property of the replication mode being
tested (see "Why the laptop cannot produce the decisive number" above).

**Why these numbers were not simply re-captured with `-nosync` added:** at
the time of this fix the same laptop was under heavy memory pressure
(~18GiB swapped, <2GiB free RAM) — conditions that would make a freshly
captured number as untrustworthy as the one being replaced, for reasons
having nothing to do with replication mode. Producing a new number under
those conditions would trade one invalid A/B for another. The corrected
procedure (this document, `-nosync` on the raft baseline) is ready to run
as soon as a machine that is not itself the bottleneck is available —
ideally directly on the dedicated box from "Running the decisive gate run"
below, skipping a second laptop pass entirely.

## Running the decisive gate run

1. Provision a dedicated, high-clock (not P/E hybrid) box — the same class of
   hardware the `netkv` README's Redis/Aerospike/Dragonfly numbers were
   captured on. Personal-account cloud instance, not shared/company infra.
   **Launch the raft baseline with `-nosync`** (see "Procedure" above) — do
   not repeat the durability-mismatch mistake this document previously made.
2. Run the load generator from a **separate** machine, or pin it to cores the
   servers do not use (`taskset`/`cgroup`), so client and server never
   contend for the same core.
3. Use production-realistic shard count (`GOMAXPROCS`-sized, per
   `cluster.Config.NumShards`'s tuning doc) rather than this doc's small `8`
   (chosen only to fit a 3-process laptop bringup).
4. Run each `(mode, conns)` point for ≥30s after a ≥5s warmup, repeat 3x, and
   report the median run — single-run laptop numbers are not sufficient
   evidence for the gate.
5. Record the table above (ops/s, p50/p99/p999 @ 128 and 256 conns, both
   modes) in this file, replacing (not deleting — keep for provenance) the
   laptop numbers, and state the gate verdict explicitly: **PASS** (PB beats
   Raft — proceed to failover) or **FAIL** (ship `pb` experimental/off, no
   further primary-backup work).

## COMMIT-MASTER vs COMMIT-ALL, and RF=2 vs Aerospike (2026-08-05, 12-vCPU box)

The comparison this file kept deferring: what the commit posture actually buys,
measured for BOTH engines on the same box, at a replication factor where the two
are contractually comparable.

**Why RF=2 is the honest shape.** At RF=2 a Raft majority is 2-of-2 and PB's full
ISR is also 2-of-2, and Aerospike's `COMMIT_ALL` likewise waits for its one
replica — every engine waits for every copy. The full-ISR-vs-majority asymmetry
that makes RF=3 flattering to Raft on loopback simply does not exist here. RF=3
PB, by contrast, waits 3-of-3 against Raft's 2-of-3.

Topology: 3 co-located nodes + generator on one 12-vCPU EPYC-Genoa box,
`GOMAXPROCS=3` per node, 8 shards, PUT-only, 100k keys, 256 B values, 15 s
measured after 5 s warmup, **0 errors on all 24 runs**, n=2 reps (agreeing within
±4%). Rostam ran `-replication-mode pb -min-isr 2`; Aerospike CE 8.1.2.4 ran a
3-node cluster with `replication-factor 2`, in-memory namespace.

| config | 8 conns | 32 conns | 128 conns |
|---|--:|--:|--:|
| Rostam commit-all (`-pb-commit-primary=false`) | 36.0k | 72.3k | 121.8k |
| Rostam commit-master (`-pb-commit-primary`) | 56.7k (**1.57x**) | 86.5k (1.20x) | 151.0k (1.24x) |
| Aerospike commit-all (`COMMIT_ALL`) | 36.4k | 51.0k | 81.5k |
| Aerospike commit-master (`COMMIT_MASTER`) | 57.6k (1.58x) | 84.6k (1.66x) | 107.1k (1.31x) |

**Two findings.**

1. **`-pb-commit-primary` works, and buys what Aerospike's commit-master buys.**
   1.57x throughput and 1.63x better p50 at 8 conns, against Aerospike's 1.58x /
   1.70x. The flag's own doc says "no throughput change on a pipelined path" —
   that is right for a saturated pipelined client, but this 1-op-per-conn
   generator converts the saved ack round trip directly into throughput, which is
   why the gain shows up here. Treat the number as a LATENCY result.
2. **At matched commit semantics Rostam leads at concurrency and ties at low
   conns.** commit-all: tie at 8, **+42% / +49%** at 32/128. commit-master: tie at
   8, **+2% / +41%**. Rostam's p50 at 128 conns is ~1.5x better in both postures.

**Caveats.** Co-located loopback, so both engines' commit-master gain is
UNDERSTATED (the saved round trip is nearly free here and would be larger on a
real network). The generator shares the box, so per the 2026-07-23 two-generator
probe these are not saturation ceilings for either engine. `netkv` cannot label
Rostam's commit posture in its output line (`-sync` is ignored for the rostam
engine — the posture is a server-side launch flag), so the two Rostam rows must be
tracked out-of-band by which flag the server was started with.

**Method note, learned the hard way.** An earlier pass of this same matrix showed
`-pb-commit-primary` doing nothing at all, which nearly motivated a pointless
rework of the `writeMu` ship path. The cause was launching a second benchmark run
while the first was still running: two benchmarks contending on 12 vCPU dropped
every baseline ~50% (Rostam commit-all @128: 81k contended vs 122k clean) and
buried the commit-master delta in noise. Confirm the previous run has exited
before starting another.

**Reproducing the Aerospike side.** `netkv/aerospike-mem.conf` in rostam-bench is
`replication-factor 1` with `address local` and no mesh seeds — a single-node
island, and at RF=1 `COMMIT_ALL` and `COMMIT_MASTER` are INDISTINGUISHABLE, so
running the matrix against it silently yields two identical Aerospike columns. A
3-node RF=2 config has to be written: three containers on `--network host` with
port triples (3000/3001/3002, 3010/../3012, 3020/../3022), every node listing all
three `mesh-seed-address-port` entries, and `replication-factor 2`. Aerospike's
parser rejects single-line brace blocks. Always assert `cluster_size=3` and
`effective_replication_factor=2` via `asinfo` before trusting a number. Bind
`address 127.0.0.1` — Aerospike CE has no auth, and a benchmark host is often
firewall-less.

## Other KV engines at a replica-ack barrier (2026-08-05, same box, n=2)

Same box, same generator, same PUT-only 100k x 256 B shape as the RF=2 table
above. Every Redis-protocol engine ran master + one replica with `netkv -sync`,
i.e. `SET` followed by `WAIT 1` — a real replica-ack barrier, and a write that
misses it is counted as an ERROR so an under-replicated run cannot masquerade as
fast (0 errors on every run below). Replication was asserted before the sweep
(`role:master connected_slaves:1` on all four).

| engine (replica-ack) | 8 conns | 32 conns | 128 conns | p50 @128 |
|---|--:|--:|--:|--:|
| **Rostam PB RF=2 commit-all** | **36.0k** | **72.3k** | **121.8k** | **947 µs** |
| Aerospike RF=2 `COMMIT_ALL` | 36.4k | 51.0k | 81.5k | 1406 µs |
| Redis 7 (`WAIT 1`) | 31.0k | 41.4k | 40.7k | 3174 µs |
| KeyDB (`WAIT 1`) | 26.2k | 42.9k | 39.7k | 3194 µs |
| Valkey 8 (`WAIT 1`) | 27.8k | 39.5k | 37.7k | 3368 µs |
| _memcached (NO replication)_ | _91.5k_ | _209.0k_ | _280.6k_ | _378 µs_ |

Rostam leads every replicated engine, and the margin widens with concurrency:
1.16x over Redis at 8 conns, **1.75x at 32, 2.99x at 128**, with p50 ~3.3x
better at 128.

**Read these two rows differently.**

- **memcached is not a peer.** It has no replication of any kind, so its numbers
  are a no-durability, no-replication ceiling, not a comparable result. It is
  listed only to show what the same harness and box produce with the barrier
  removed entirely.
- **Dragonfly is EXCLUDED, not lost.** It reported 78 / 313 / 1245 ops/s with p50
  pinned at ~101.9 ms at every connection count. A latency that is constant
  regardless of offered load is a fixed interval, not a throughput limit: it is
  Dragonfly's `WAIT` replication-ack granularity (~100 ms), so the harness was
  measuring that timer rather than the engine. Reporting it as a ~1000x deficit
  would be wrong. Comparing Dragonfly here needs a different ack mechanism.

**Caveat — this is system-vs-system at equal replication, NOT per-core
efficiency.** Rostam ran 3 nodes at `GOMAXPROCS=3` (up to 9 cores); Redis and
Valkey are single-threaded per instance and use ~1-2 cores; KeyDB and Dragonfly
are multi-threaded. A per-core comparison would be a different measurement and
would narrow the gap substantially. The generator also shares the box, so none of
these are saturation ceilings. ClickHouse belonging to another workload was using
~30% of one core throughout.

## Snapshot-transfer write stall (measured)

Snapshot transfer serializes the whole FSM inside `Engine.RunExclusive`'s
critical section (`writeMu` + `e.mu`), which excludes **both** `Applier.Apply`
sites. The shard's write path is therefore frozen for the duration of the
serialization. This is flow-control **option (a)** — accepted deliberately, with
the cost named rather than hidden. The alternatives and why they were rejected
are documented on `Engine.takeSnapshot`; the short form:

- **(b) freeze-under-lock, serialize off-lock** is not reachable today. The
  cache half has a plausible freeze point (mmap page-append with a
  `rebuildIndexFromPages` recovery path), but the vector half does not:
  `CollectionStore.SnapshotAll` walks HNSW graphs mutated in place under the
  collection lock, with no versioned or copy-on-write view to pin. Freezing only
  the cache yields a blob whose two halves sit at different logical points —
  exactly the torn snapshot `RunExclusive` exists to prevent.
- **(c) ship from the on-disk backup artifact** has zero write-path impact but
  makes convergence strictly *worse* where it matters: an artifact's frontier
  `F` is older, so `F+1` is further outside the primary's catch-up ring and the
  re-snapshot loop is *more* likely to spin.

### Numbers

`go test ./shard/ -run xxx -bench PBSnapshotStall -benchtime 3x`
(i9-13900H, single PB shard, mmap cache, 100-byte values, **KV only — no vector
collections**):

| keys | blob     | stall     |
|-----:|---------:|----------:|
|  1e3 | ~0.12 MB |   0.68 ms |
|  1e4 |  ~1.2 MB |   2.70 ms |
|  1e5 |   ~12 MB |   31.2 ms |
|  5e5 |   ~60 MB |  134.9 ms |

Throughput is ~450 MB/s and **linear in serialized bytes**, so:

    stall ≈ shard_bytes / 450 MB/s

### Operational ceiling

A shard whose *total serialized* size exceeds **~450 MB** stalls its write path
for over a second per transfer. Split shards above that. Vector collections
serialize through the same critical section and add to the blob, so size against
total serialized bytes, not the KV set alone.

`Engine.SnapshotStats().StallMaxNs` reports the live per-shard figure. Alert on
it directly rather than inferring the stall from key counts — it is the only
measurement that includes this deployment's vector state.

### What mitigates it

The case that motivates the whole stage — a `minISR>=2` shard left write-dead by
a failover — is *already* refusing writes (`ErrBelowMinISR`), so there is no
write traffic to stall. The stall only bites on a healthy shard growing a
deeply-lagged member, which is also the case where the ring delta usually serves
the grow without a snapshot at all.
