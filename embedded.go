// SPDX-License-Identifier: Apache-2.0

package rostam

import (
	"cmp"
	"context"
	"crypto/tls"
	"encoding/binary"
	"errors"
	"fmt"
	"log/slog"
	"slices"
	"sort"
	"strings"
	"sync"
	"time"

	"github.com/rostamlabs/rostam/cache"
	"github.com/rostamlabs/rostam/cluster"
	"github.com/rostamlabs/rostam/ops"
	"github.com/rostamlabs/rostam/shard"
	"github.com/rostamlabs/rostam/vector"
)

// EmbeddedConfig configures an in-process Rostam node.
type EmbeddedConfig struct {
	// NodeID is the unique identifier for this node in the Raft cluster.
	NodeID string

	// DataDir is the base directory for Raft logs, snapshots, and mmap files.
	DataDir string

	// NumShards is the number of independent Raft shards. Defaults to 64
	// when zero. See cluster.Config.NumShards for the tuning rationale (fewer
	// groups = higher write throughput on commodity nodes; raise for many
	// large-core nodes). Fixed at creation.
	NumShards int

	// ReplicationFactor is how many nodes host each shard. 0 (default) or
	// >= len(Peers) means full replication (every node hosts every shard); a
	// smaller value partitions shards across the cluster — each node stores only
	// its shards and forwards ops for others to an owner.
	ReplicationFactor int

	// Peers is the static cluster membership list. nil or empty means
	// single-node mode.
	Peers []Peer

	// Bootstrap controls whether the node bootstraps itself as a fresh
	// single-node cluster. Set true on first start only.
	Bootstrap bool

	// Ops is the registry of named operations. Must be non-nil.
	Ops *ops.Registry

	// Cache configures cache-layer behaviour (durability, mlock, msync).
	Cache CacheConfig

	// RaftAddr is this node's multiplexed Raft transport endpoint.
	// Required when len(Peers) > 1.
	RaftAddr string

	// RaftTransport selects the inter-node Raft transport: "" or "mux" (default)
	// uses the per-group NetworkTransport over the shared TCP listener; "fabric"
	// uses the multiplexed batching transport (raft/fabric). Flag-gated so the
	// default path is unchanged.
	RaftTransport string

	// RaftHeartbeatMs overrides the Raft heartbeat interval in milliseconds.
	// Zero uses the shard package default.
	RaftHeartbeatMs int

	// RaftElectionMs overrides the Raft election timeout in milliseconds.
	// Zero uses the shard package default.
	RaftElectionMs int

	// NoSync disables fsync on Raft log writes. Improves throughput in
	// testing; do not use in production.
	NoSync bool

	// VolatileLog puts data shards' Raft logs fully in memory (no write()
	// syscall); durability comes only from replication. The meta group stays
	// durable. See raft.Config.VolatileLog for the fresh-rejoin safety contract.
	VolatileLog bool

	// RaftLogLevel controls how loud the embedded hashicorp/raft is:
	// "TRACE"|"DEBUG"|"INFO"|"WARN"|"ERROR"|"OFF". Empty means INFO. Set
	// "ERROR" in tests and benchmarks that build clusters in a loop — `go test`
	// merges the test binary's stdout and stderr, so raft's per-election output
	// otherwise interleaves with the results.
	RaftLogLevel string

	// PersistentVectors makes vector collections mmap-backed (off-heap) on every
	// node. Raft stays the durability authority (vectors are wiped at startup and
	// repopulated from the Raft snapshot/log); this only changes the in-memory
	// layout, trading heap/GC pressure for the OS page cache. Recommended for
	// large vector datasets.
	PersistentVectors bool

	// PBFrontierStampInterval bounds how often the PB applied frontier is persisted
	// into the cache header. 0 selects the shard default (1s).
	//
	// It is exposed here because the cost is a full-region msync PER SHARD per
	// tick, so it is the knob that governs write TAIL latency in PB mode — and it
	// was previously unreachable: shard.Config carried it, but nothing plumbed it,
	// so no embedded caller or operator could change it. Lowering it tightens a
	// restarted node's catch-up delta at the cost of that tail; raising it does the
	// reverse. Neither direction affects correctness (see shard.Config's note: a
	// staler watermark only under-reports, and log matching turns an under-report
	// into a true-prefix catch-up or a clean divergence reject).
	//
	// PBFrontierStampEvery is the OPT-IN write-count trigger, off by default
	// because an msync costs O(mapped region) rather than O(bytes changed) — see
	// shard's defaultPBFrontierStampEvery for the measured penalty.
	PBFrontierStampInterval time.Duration
	PBFrontierStampEvery    int

	// InternalToken is the inter-node service credential presented by every
	// forwarding/admin client this node opens to a peer. The destination node's
	// RBAC authorizer treats it as the superuser service principal so an inter-node
	// forward passes auth. REQUIRED when the cluster runs with RBAC enabled
	// (otherwise a write forwarded from a non-leader node arrives token-less and is
	// denied). Empty = no token (correct for nil-auth / open clusters). Inter-node
	// traffic is plaintext this round, so this token is the only inter-node auth.
	InternalToken string

	// InterNodeTLS is the TLS CLIENT config for the inter-node forwarding dial,
	// threaded straight into cluster.Config.InterNodeTLS. Set it (alongside client
	// TLS) so peerClient dials TLS-wrapped peer ports over TLS, verifying each peer's
	// server cert against the CA. nil ⇒ plaintext inter-node dial (default; zero cost
	// when client TLS is off). AUTH is still the internal token. See
	// cluster.Config.InterNodeTLS.
	InterNodeTLS *tls.Config

	// InterNodeServerTLS is the TLS SERVER config that wraps the inter-node
	// REPLICATION listeners (Raft mux/fabric + PB), threaded straight into
	// cluster.Config.InterNodeServerTLS. Set it (alongside InterNodeTLS) so the
	// replication ports are mTLS-authenticated, not just the forwarding dial. nil ⇒
	// plaintext replication listeners (default; byte-identical to today). See
	// cluster.Config.InterNodeServerTLS.
	InterNodeServerTLS *tls.Config

	// NodeCNAllowlist is the OPT-IN per-node mTLS identity allowlist, threaded
	// straight into cluster.Config.NodeCNAllowlist (the inter-node CLIENT peer-CN
	// verify). Empty/nil = OFF = byte-identical (no callback attached). See
	// cluster.Config.NodeCNAllowlist.
	NodeCNAllowlist map[string]bool

	// ReplicationMode selects the cluster-level data-plane replication engine,
	// threaded straight into cluster.Config.ReplicationMode: "" or "raft"
	// (default) uses per-shard Raft groups, byte-identical to today. "pb"
	// selects EXPERIMENTAL primary-backup/ISR replication (shard.ReplicationModePB)
	// for every shard — a static cluster only (no automatic failover yet; see
	// shard/pbisr/BENCHMARK.md for the measured comparison). Requires
	// MinISR >= 1 and every Peer's PBAddr set.
	ReplicationMode string

	// MinISR is the minimum in-sync-replica count required by "pb" mode (must be
	// >= 1 when ReplicationMode == "pb"). Unused in "raft"/"" mode. Threaded
	// straight into cluster.Config.MinISR.
	MinISR int

	// PBCommitPrimary selects commit-on-primary durability for "pb" mode
	// (threaded into cluster.Config.PBCommitPrimary): the primary acks on local
	// apply and replicates asynchronously. DURABILITY DOWNGRADE — an acked write
	// can be lost if the primary dies before a backup received it. Default false
	// waits for the full ISR. Unused in "raft"/"" mode.
	PBCommitPrimary bool

	// PBAutoFailover enables automatic primary-backup failover for "pb"
	// mode (threaded into cluster.Config.PBAutoFailover): each primary commits a
	// liveness beacon, the meta leader promotes an ISR survivor when one goes
	// silent, and the ISR shrink/grow drivers un-wedge and re-open shards. Without
	// it a PB shard whose primary dies stays DOWN until an operator intervenes.
	// Off (false) is byte-identical to the static pre-Plan-4 cluster: no beacon
	// reaches the meta-Raft log and no epoch is ever bumped automatically.
	//
	// NOTE the default differs by entry point: rostam-server defaults
	// -pb-auto-failover to TRUE (both pre-default-on gates pass — see
	// shard/pbisr/DESIGN.md), whereas this field's Go zero value
	// is false, so a direct library embedder must opt in explicitly. Unused in
	// "raft"/"" mode.
	PBAutoFailover bool

	// WASMBlobRetention enables WASM blob retirement (threaded straight into
	// cluster.Config.WASMBlobRetention): how long a module blob that nothing on
	// this node references — a superseded version, or a __wasm_blob_put__ orphan —
	// is kept before its file is deleted.
	//
	// ZERO (THE DEFAULT, AND THE DEFAULT ON rostam-server TOO) DISABLES IT: no
	// sweeper runs and nothing is ever removed. Read cluster.Config's field
	// documentation and cluster/wasm_blob_retire.go before setting it — the value
	// is an assertion about how far behind a replica may fall, not a tuning knob,
	// and getting it wrong parks a lagging replica until an operator supplies the
	// bytes by hand with __wasm_blob_put__.
	WASMBlobRetention time.Duration
}

// ReshardState is the embedded-level view of a collection's online-reshard
// status. Status 0 = Stable (no reshard); 1 = Resharding (a live repartition is
// dual-writing to the new gen). OldP/OldGen mirror the live PartitionsGen at
// reshard-begin (the read source of truth during the reshard); NewP/NewGen are
// the target gen being copied into. The zero value is Stable.
//
// Exported because the dual-write routing and the reshard orchestrator
// consume it across embedded.go, and a future reshard-progress query
// API may surface it on the public Store interface.
type ReshardState struct {
	Status uint8
	OldP   int
	OldGen uint32
	NewP   int
	NewGen uint32
}

// AliasAction is one mutation in an atomic alias batch at the embedded layer. It
// mirrors cluster.AliasAction (the meta-Raft wire type) so the partitionCatalog
// interface stays decoupled from the cluster package — metaCatalog bridges the
// two, exactly as ReshardState bridges to cluster.ReshardEntry. Names are
// canonicalized by the catalog before they reach meta-Raft. Delete=true removes
// the alias; otherwise Alias is created/overwritten to point at Canonical.
type AliasAction struct {
	Alias     string
	Canonical string
	Delete    bool
}

// partitionCatalog is the minimal partition-count lookup embedded needs. Both
// the in-process ops.Catalog-backed single-node wrapper and metaCatalog
// (multi-node) satisfy it, so the partitioned-routing call sites are
// backend-agnostic.
type partitionCatalog interface {
	PartitionsGen(collection string) (p int, gen uint32, ok bool)
	SetPartitionsGen(collection string, p int, gen uint32) error
	// ReshardState reports a collection's online-reshard state. ok=false means
	// Stable / no reshard in progress.
	ReshardState(collection string) (st ReshardState, ok bool)
	// SetReshardState records a collection's online-reshard state. A st.Status of
	// Stable (0) clears any in-progress reshard.
	SetReshardState(collection string, st ReshardState) error
	// ResolveAlias returns the canonical target collection an alias resolves to.
	// ok=false means the name is not an alias (one level only; targets are real
	// collections). The input name is canonicalized before lookup.
	ResolveAlias(name string) (canonical string, ok bool)
	// SetAliases atomically applies a batch of alias mutations. All names are
	// canonicalized before being committed.
	SetAliases(actions []AliasAction) error
	// ListAliases returns a snapshot of the alias map (alias→target), both
	// canonical.
	ListAliases() map[string]string
}

// metaCatalog is the multi-node durable catalog backed by the meta-Raft FSM.
// Reads are local (no network); writes go through consensus with leader-forward.
// Partitions reflects this node's locally-applied catalog state and may briefly
// lag for a collection created on another node: during that convergence window
// (one replication round-trip) searches route to the empty logical collection
// and return empty results until this node applies the entry.
type metaCatalog struct {
	node *cluster.Node
}

// PartitionsGen reads the collection's partition count and generation from the
// local meta-Raft FSM (no network). It canonicalizes the name so the key matches
// what SetPartitionsGen wrote. A missing entry or P<=1 reports (1, 0, false):
// single-partition default.
func (m *metaCatalog) PartitionsGen(collection string) (int, uint32, bool) {
	p, gen, ok := m.node.CollectionPartitionsGen(ops.CanonicalName(collection))
	if !ok || p <= 1 {
		return 1, 0, false
	}
	return int(p), gen, true
}

// SetPartitionsGen commits the collection's partition count and generation
// through meta-Raft consensus (leader-forwarding on a follower), bounded by a
// 5s timeout.
func (m *metaCatalog) SetPartitionsGen(collection string, p int, gen uint32) error {
	return m.node.SetCollectionPartitions(ops.CanonicalName(collection), uint32(p), gen, 5*time.Second)
}

// ReshardState reads the collection's online-reshard state from the local
// meta-Raft FSM (no network). OldP/OldGen come from the SOURCE pin stored in the
// reshard entry (NOT the live PartitionsGen — after the Phase-4 cutover the live
// PartitionsGen IS the new gen, so deriving Old from it would point both dual-
// write legs at the new gen and silently stop writing the old gen, the bug this
// fixes); NewP/NewGen come from the reshard entry's target. ok=false (Stable) when
// no reshard is in progress.
//
// Backward-compat: a reshard entry persisted before the Source fields existed
// (e.g. an old meta snapshot, or a reshard begun pre-upgrade) decodes
// SourceP=SourceGen=0; dualTargets then falls back to today's collapse-at-cutover
// behavior for that one in-flight reshard.
func (m *metaCatalog) ReshardState(collection string) (ReshardState, bool) {
	canon := ops.CanonicalName(collection)
	e, ok := m.node.CollectionReshard(canon)
	if !ok || e.Status == 0 {
		return ReshardState{}, false
	}
	return ReshardState{
		Status: e.Status,
		OldP:   e.SourceP,
		OldGen: e.SourceGen,
		NewP:   e.TargetP,
		NewGen: e.TargetGen,
	}, true
}

// SetReshardState commits the collection's online-reshard state through meta-Raft
// consensus (leader-forwarding on a follower). Status/NewP/NewGen AND the
// OldP/OldGen source pin are all persisted, so the dual-write keeps hitting the
// old gen after the cutover flips the live PartitionsGen. A Stable status clears
// the reshard entry.
func (m *metaCatalog) SetReshardState(collection string, st ReshardState) error {
	return m.node.SetCollectionReshard(ops.CanonicalName(collection), cluster.ReshardEntry{
		Status:    st.Status,
		TargetP:   st.NewP,
		TargetGen: st.NewGen,
		SourceP:   st.OldP,
		SourceGen: st.OldGen,
	}, 5*time.Second)
}

// ResolveAlias reads the alias→target mapping from the local meta-Raft FSM (no
// network). The input is canonicalized so the key matches what SetAliases wrote
// (the catalog stores canonical→canonical). ok=false means not an alias.
func (m *metaCatalog) ResolveAlias(name string) (string, bool) {
	return m.node.ResolveAlias(ops.CanonicalName(name))
}

// SetAliases commits a batch of alias mutations through meta-Raft consensus as
// ONE atomic log entry (leader-forwarding on a follower), bounded by a 5s
// timeout. Both the alias name and the target are canonicalized so the catalog
// stores canonical→canonical.
func (m *metaCatalog) SetAliases(actions []AliasAction) error {
	out := make([]cluster.AliasAction, len(actions))
	for i, a := range actions {
		out[i] = cluster.AliasAction{
			Alias:     ops.CanonicalName(a.Alias),
			Canonical: ops.CanonicalName(a.Canonical),
			Delete:    a.Delete,
		}
	}
	return m.node.SetAliases(out, 5*time.Second)
}

// ListAliases returns a snapshot of the alias map (canonical→canonical) from the
// local meta-Raft FSM (no network).
func (m *metaCatalog) ListAliases() map[string]string {
	return m.node.ListAliases()
}

// singleNodeCatalog wraps the in-process ops.Catalog (single-node deployments,
// no meta-Raft) to satisfy partitionCatalog including the reshard methods. The
// ops.Catalog already exposes PartitionsGen/SetPartitionsGen and the lower-level
// ReshardGen/SetReshardGen; this adapter maps the latter to the embedded-level
// ReshardState (deriving OldP/OldGen from the live PartitionsGen).
type singleNodeCatalog struct {
	cat *ops.Catalog

	// node is the durable shard KV (NumShards:1, Bootstrap). The alias map is
	// persisted under the reserved __vcat__/aliases key via the same raw put/get
	// ops the partition catalog uses, so aliases survive a restart (Raft → WAL/
	// snapshot). It is also used for the lazy startup load of the alias cache.
	node *cluster.Node

	// aliases holds the single-node alias catalog (canonical→canonical) as an
	// in-memory READ CACHE guarded by aliasMu, kept in sync with the durable KV.
	// It is WRITE-THROUGH (SetAliases persists the whole map to __vcat__/aliases
	// before updating the cache) and LOADED at first use from that durable key
	// (loadOnce), so the cache and the persisted state never diverge and aliases
	// are durable across a node restart. The batch applies under a single
	// aliasMu.Lock() so a swap is atomic with respect to concurrent reads, and the
	// durable write is ONE KV put of the whole map (atomic-swap semantics mirroring
	// meta-Raft: a restart never observes a half-applied alias set).
	//
	// An RWMutex keeps the data-plane read path (resolveAlias on every op) off a
	// write-lock — concurrent resolves take RLock and never serialize against each
	// other; only a swap (SetAliases) takes the exclusive Lock.
	aliasMu  sync.RWMutex
	aliases  map[string]string
	loadOnce sync.Once
}

// aliasCatalogKey is the reserved durable KV key holding the serialized single-
// node alias map. It rides the existing __vcat__/ catalog-metadata namespace
// (plain KV, NOT partition-routed) so it survives a restart via the shard Raft
// without any new on-disk format. It is a raw key (not catalogKey-derived: that
// helper namespaces a COLLECTION; this is the singleton alias map).
var aliasCatalogKey = []byte("__vcat__/aliases")

// ensureAliasesLoaded lazily loads the alias map from the durable KV into the
// in-memory cache exactly once. It is called on the first alias read/write, by
// which point the shard Raft leader is elected (every caller — data-plane ops via
// resolveAlias, or the alias admin ops — runs after startup, and the durable get
// itself requires a ready leader). Loading lazily (rather than blocking
// NewEmbedded on leader election) avoids the bootstrap chicken-and-egg: the
// catalog is never read before the shard KV is ready, and the get goes straight
// to the shard (no catalog-routing recursion). A missing key (fresh node) loads
// an empty map.
func (s *singleNodeCatalog) ensureAliasesLoaded() {
	s.loadOnce.Do(func() {
		m := map[string]string{}
		if s.node != nil {
			if raw, err := s.node.Call("get", ops.EncodeKeyArgs(aliasCatalogKey)); err == nil && len(raw) > 0 {
				if decoded, ok := decodeAliasMap(raw); ok {
					m = decoded
				}
			}
		}
		s.aliasMu.Lock()
		if s.aliases == nil {
			s.aliases = m
		}
		s.aliasMu.Unlock()
	})
}

// persistAliasesLocked writes the whole in-memory alias map to the durable KV as
// ONE put (atomic swap). Caller holds aliasMu. Returns an error if the durable
// write fails, so SetAliases does not ack a non-durable change.
func (s *singleNodeCatalog) persistAliasesLocked() error {
	if s.node == nil {
		return nil
	}
	_, err := s.node.Call("put", ops.EncodePutArgs(aliasCatalogKey, encodeAliasMap(s.aliases), 0))
	return mapErr(err)
}

// encodeAliasMap serializes an alias map as
// [count u32]{[aliasLen u32][alias][targetLen u32][target]}... (little-endian).
func encodeAliasMap(m map[string]string) []byte {
	size := 4
	for k, v := range m {
		size += 8 + len(k) + len(v)
	}
	b := make([]byte, size)
	binary.LittleEndian.PutUint32(b[0:4], uint32(len(m)))
	off := 4
	for k, v := range m {
		binary.LittleEndian.PutUint32(b[off:off+4], uint32(len(k)))
		off += 4
		copy(b[off:], k)
		off += len(k)
		binary.LittleEndian.PutUint32(b[off:off+4], uint32(len(v)))
		off += 4
		copy(b[off:], v)
		off += len(v)
	}
	return b
}

// decodeAliasMap reverses encodeAliasMap. Returns ok=false on a malformed/short
// buffer (caller treats it as an empty map rather than panicking).
func decodeAliasMap(b []byte) (map[string]string, bool) {
	if len(b) < 4 {
		return nil, false
	}
	n := int(binary.LittleEndian.Uint32(b[0:4]))
	off := 4
	m := make(map[string]string, n)
	for i := 0; i < n; i++ {
		if off+4 > len(b) {
			return nil, false
		}
		kl := int(binary.LittleEndian.Uint32(b[off : off+4]))
		off += 4
		if off+kl > len(b) {
			return nil, false
		}
		k := string(b[off : off+kl])
		off += kl
		if off+4 > len(b) {
			return nil, false
		}
		vl := int(binary.LittleEndian.Uint32(b[off : off+4]))
		off += 4
		if off+vl > len(b) {
			return nil, false
		}
		v := string(b[off : off+vl])
		off += vl
		m[k] = v
	}
	return m, true
}

func (s *singleNodeCatalog) PartitionsGen(collection string) (int, uint32, bool) {
	return s.cat.PartitionsGen(collection)
}

func (s *singleNodeCatalog) SetPartitionsGen(collection string, p int, gen uint32) error {
	return s.cat.SetPartitionsGen(collection, p, gen)
}

// ReshardState maps the record's status/target/source fields to the embedded
// view: OldP/OldGen are the SOURCE gen pinned at reshard-begin (NOT the live
// PartitionsGen — after the cutover flip the live gen IS the new gen, so deriving
// Old from it would point both legs at the new gen and silently drop the old gen);
// NewP/NewGen are the record's TargetP/TargetGen. ok=false when Stable.
//
// Backward-compat: a Resharding record written before the Source fields existed
// reports OldP=OldGen=0; dualTargets then falls back to today's collapse-at-
// cutover behavior for that one in-flight reshard.
func (s *singleNodeCatalog) ReshardState(collection string) (ReshardState, bool) {
	status, targetP, targetGen, sourceP, sourceGen, ok := s.cat.ReshardGen(collection)
	if !ok || status == 0 {
		return ReshardState{}, false
	}
	return ReshardState{
		Status: uint8(status),
		OldP:   int(sourceP),
		OldGen: sourceGen,
		NewP:   int(targetP),
		NewGen: targetGen,
	}, true
}

// SetReshardState persists status + target gen + source (old) gen on the record.
// A Stable status clears the reshard fields (writing back the legacy 12-byte
// form). OldP/OldGen are now PERSISTED (the Source pin) so the dual-write keeps
// hitting the old gen after the cutover flips the live PartitionsGen.
func (s *singleNodeCatalog) SetReshardState(collection string, st ReshardState) error {
	return s.cat.SetReshardGen(collection, uint32(st.Status), uint32(st.NewP), st.NewGen, uint32(st.OldP), st.OldGen)
}

// ResolveAlias reads the in-memory alias map (canonical→canonical). The input is
// canonicalized so the key matches what SetAliases stored. ok=false means the
// name is not an alias.
func (s *singleNodeCatalog) ResolveAlias(name string) (string, bool) {
	s.ensureAliasesLoaded()
	s.aliasMu.RLock()
	defer s.aliasMu.RUnlock()
	canonical, ok := s.aliases[ops.CanonicalName(name)]
	return canonical, ok
}

// SetAliases applies the whole batch under a single lock (atomic swap) and
// persists the resulting map durably as ONE KV put (so a restart never observes a
// half-applied set — mirroring meta-Raft atomic-swap semantics). Both alias and
// target are canonicalized so the catalog stores canonical→canonical. The cache
// is WRITE-THROUGH: the durable put happens before the lock is released, and on a
// durable-write failure the in-memory map is rolled back so the cache never
// diverges from the persisted state.
func (s *singleNodeCatalog) SetAliases(actions []AliasAction) error {
	s.ensureAliasesLoaded()
	s.aliasMu.Lock()
	defer s.aliasMu.Unlock()
	if s.aliases == nil {
		s.aliases = make(map[string]string)
	}
	// Snapshot the prior map so a failed durable write can roll back the cache.
	prev := make(map[string]string, len(s.aliases))
	for k, v := range s.aliases {
		prev[k] = v
	}
	for _, a := range actions {
		alias := ops.CanonicalName(a.Alias)
		if a.Delete {
			delete(s.aliases, alias)
		} else {
			s.aliases[alias] = ops.CanonicalName(a.Canonical)
		}
	}
	if err := s.persistAliasesLocked(); err != nil {
		s.aliases = prev
		return err
	}
	return nil
}

// ListAliases returns a snapshot copy of the in-memory alias map (loaded from the
// durable KV on first use, then kept write-through).
func (s *singleNodeCatalog) ListAliases() map[string]string {
	s.ensureAliasesLoaded()
	s.aliasMu.RLock()
	defer s.aliasMu.RUnlock()
	out := make(map[string]string, len(s.aliases))
	for k, v := range s.aliases {
		out[k] = v
	}
	return out
}

// embedded is the in-process Store backend backed by a cluster.Node.
type embedded struct {
	node *cluster.Node

	// lf follows a hosted-shard NotLeader result to the leader server-side, and
	// is what e.Call dispatches through.
	//
	// Node.Call forwards an op for a shard this node does NOT host, but a shard it
	// hosts as a FOLLOWER/BACKUP comes back as a NotLeaderError instead. The
	// binary TCP client follows that hint itself; nothing inside this package did,
	// so every internal coordinator — notably the partitioned-create loop below —
	// aborted whenever a partition happened to land on a shard this node backs.
	// That contradicted the assumption recorded at cluster.Node.CallPhysical
	// ("OpReadWrite ops already route to the leader resiliently"), which holds
	// only for the not-hosted case.
	//
	// This is the SAME wrapper the HTTP and gRPC transports already use, not a new
	// forwarding path: one hop to the hinted leader, and the per-peer client
	// follows any further hops. It is only effective because the hint is now
	// populated under PB as well as Raft (see cluster.resolveLeaderHint) — before
	// that fix LeaderAddr was empty in PB mode and this wrapper silently did
	// nothing there.
	lf *cluster.LeaderFollowingDispatcher

	// catalog records the partition count per collection. It is consulted on the
	// vector point/search paths to decide whether to route/fan-out. The single-
	// partition path never writes the catalog, so an unpartitioned collection has
	// no entry and the lookup is a miss (a cheap local shard "get", no fan-out).
	//
	// Backend selection (see NewEmbedded): multi-node uses metaCatalog (durable,
	// meta-Raft-backed, visible on every node); single-node uses the in-process
	// ops.Catalog over kvCatalogStore — the node's OWN durable shard KV over the
	// reserved __vcat__/ namespace, so single-node partition counts are now durable
	// across a restart too (there is no meta-Raft in single-node mode). Both satisfy
	// partitionCatalog, so every e.catalog.* call site below is backend-agnostic.
	//
	// Plan-1 scope: the cross-shard fan-out built on this catalog lives only in the
	// embedded backend. The remote APIs (httpapi/grpcapi/client) are NOT
	// partition-aware, so partitioned collections (Partitions>1) must be driven
	// through the embedded backend until the networked-coordinator path lands.
	catalog partitionCatalog
}

// kvCatalogStore is the single-node CatalogStore backing the partition catalog
// with the node's OWN durable shard KV (NumShards:1, Bootstrap) over the reserved
// "__vcat__/" key namespace (catalogKey already prefixes it). It replaces the old
// in-memory map so partition counts are DURABLE across a restart: a PutCatalog
// goes through the shard's Raft "put" op (Raft → WAL/snapshot → crash-consistent
// + replayed on restart) and GetCatalog reads it back via the "get" op.
//
// It calls node.Call("put"/"get") DIRECTLY — these are plain key/value ops, NOT
// partition-routed (the __vcat__/ keys are catalog metadata, never vector data),
// so there is no recursion back through the partition catalog (no chicken-and-egg
// at startup). Reads/writes require a ready shard leader; every caller runs after
// startup (CreateCollection + the data-plane routing reads), so the catalog is
// never consulted before the shard KV is up. The partition catalog has no
// in-process cache (ops.Catalog reads through on every call), so there is nothing
// to load at startup — the durable KV IS the source of truth.
type kvCatalogStore struct {
	node *cluster.Node
}

func newKVCatalogStore(node *cluster.Node) *kvCatalogStore {
	return &kvCatalogStore{node: node}
}

// GetCatalog reads key from the durable shard KV. A missing key (or any read
// error — e.g. a not-yet-ready leader) reports (nil, false): the catalog treats
// it as "no entry", i.e. the single-partition default.
func (s *kvCatalogStore) GetCatalog(key []byte) ([]byte, bool) {
	raw, err := s.node.Call("get", ops.EncodeKeyArgs(key))
	if err != nil || len(raw) == 0 {
		return nil, false
	}
	return raw, true
}

// PutCatalog writes key/val durably through the shard's Raft "put" op (TTL 0 =
// no expiry). It returns the underlying error so SetPartitions does not ack a
// non-durable write.
func (s *kvCatalogStore) PutCatalog(key, val []byte) error {
	_, err := s.node.Call("put", ops.EncodePutArgs(key, val, 0))
	return mapErr(err)
}

// NewEmbedded constructs an in-process Store from cfg. cfg.Ops must be
// non-nil. NumShards defaults to 64 when zero.
// clusterConfigFrom translates an EmbeddedConfig into the cluster.Config the node
// is built from. It is a PURE function (numShards and shardCfg are the two values
// NewEmbedded derives before this point) so the translation is unit-testable on
// its own: a knob that exists on EmbeddedConfig but is never copied here is
// silently unreachable at runtime, which is exactly how PBAutoFailover shipped
// with no way to enable it from the server. TestClusterConfigFromThreadsPBKnobs
// guards that.
func clusterConfigFrom(cfg EmbeddedConfig, numShards int, shardCfg shard.Config) cluster.Config {
	return cluster.Config{
		NodeID:             cfg.NodeID,
		DataDir:            cfg.DataDir,
		NumShards:          numShards,
		ReplicationFactor:  cfg.ReplicationFactor,
		Bootstrap:          cfg.Bootstrap,
		ShardCfg:           shardCfg,
		Ops:                cfg.Ops,
		Peers:              translatePeers(cfg.Peers),
		RaftAddr:           cfg.RaftAddr,
		RaftTransport:      cfg.RaftTransport,
		InternalToken:      cfg.InternalToken,
		InterNodeTLS:       cfg.InterNodeTLS,
		InterNodeServerTLS: cfg.InterNodeServerTLS,
		NodeCNAllowlist:    cfg.NodeCNAllowlist,
		ReplicationMode:    cfg.ReplicationMode,
		PBCommitPrimary:    cfg.PBCommitPrimary,
		PBAutoFailover:     cfg.PBAutoFailover,
		MinISR:             cfg.MinISR,
		WASMBlobRetention:  cfg.WASMBlobRetention,
	}
}

func NewEmbedded(cfg EmbeddedConfig) (Store, error) {
	if cfg.Ops == nil {
		return nil, errors.New("rostam: EmbeddedConfig.Ops must not be nil")
	}

	numShards := cfg.NumShards
	if numShards == 0 {
		numShards = 64
	}

	cc := cache.DefaultConfig()
	cc.NumShards = 1 // forced per shard; cluster handles fanout
	cc.Durable = cfg.Cache.Durable
	cc.Mlock = cfg.Cache.Mlock
	cc.DisableColdCompaction = cfg.Cache.DisableColdCompaction
	if cfg.Cache.MsyncIntervalMs != 0 {
		cc.MsyncIntervalMs = cfg.Cache.MsyncIntervalMs
	}
	// Spread the node's budget across its Raft shards: this node holds
	// numShards caches, each pinned to cc.NumShards = 1, so the divisor is
	// numShards and NOT cc.NumShards. Previously each shard inherited
	// cache.DefaultConfig()'s 256 MiB per-shard cap, making the node's real
	// bound numShards * 256 MiB (16 GiB at the default 64 shards) with no way
	// to lower it.
	if err := applyCacheBudget(&cc, cfg.Cache.MaxMemoryBytes, numShards); err != nil {
		return nil, err
	}

	shardCfg := shard.DefaultConfig(cfg.DataDir, cfg.NodeID, cfg.Ops)
	shardCfg.Cache = cc
	shardCfg.RaftHeartbeatMs = cfg.RaftHeartbeatMs
	shardCfg.RaftElectionMs = cfg.RaftElectionMs
	shardCfg.NoSync = cfg.NoSync
	shardCfg.VolatileLog = cfg.VolatileLog
	shardCfg.RaftLogLevel = cfg.RaftLogLevel
	shardCfg.PersistentVectors = cfg.PersistentVectors
	shardCfg.PBFrontierStampInterval = cfg.PBFrontierStampInterval
	shardCfg.PBFrontierStampEvery = cfg.PBFrontierStampEvery

	node, err := cluster.New(clusterConfigFrom(cfg, numShards, shardCfg))
	if err != nil {
		return nil, err
	}

	// Backend selection must use the SAME multi-node predicate the cluster layer
	// uses to start meta-Raft (cluster.Config.isMultiNode == len(Peers) > 0), so
	// the catalog backend exists iff meta-Raft does: multi-node gets the durable,
	// cross-node meta-Raft catalog; single-node keeps the in-process map (there is
	// no meta-Raft to back it).
	var cat partitionCatalog
	if len(cfg.Peers) > 0 {
		cat = &metaCatalog{node: node}
	} else {
		cat = &singleNodeCatalog{cat: ops.NewCatalog(newKVCatalogStore(node)), node: node}
	}
	return &embedded{node: node, lf: node.LeaderFollowingDispatcher(), catalog: cat}, nil
}

// translatePeers converts the public Peer slice to cluster.Peer.
func translatePeers(in []Peer) []cluster.Peer {
	if len(in) == 0 {
		return nil
	}
	out := make([]cluster.Peer, len(in))
	for i, p := range in {
		out[i] = cluster.Peer{
			NodeID:     p.NodeID,
			RaftAddr:   p.RaftAddr,
			ServerAddr: p.ServerAddr,
			PBAddr:     p.PBAddr,
		}
	}
	return out
}

// Get reads the value for key from the local cache (no Raft). Returns
// ErrNotFound if the key is absent or expired.
func (e *embedded) Get(_ context.Context, key []byte) ([]byte, error) {
	result, err := e.node.Call("get", ops.EncodeKeyArgs(key))
	if err != nil {
		return nil, mapErr(err)
	}
	return result, nil
}

// GetInto is the allocation-light Get: the value is copied into dst (reusing its
// capacity when large enough). Same ErrNotFound semantics as Get; the returned
// slice may alias dst.
func (e *embedded) GetInto(_ context.Context, key, dst []byte) ([]byte, error) {
	result, err := e.node.Call("get", ops.EncodeKeyArgs(key))
	if err != nil {
		return nil, mapErr(err)
	}
	return append(dst[:0], result...), nil
}

// Put writes key/value through Raft with the given TTL. Returns
// ErrNotLeader if the shard leader is unavailable.
func (e *embedded) Put(_ context.Context, key, value []byte, ttl time.Duration) error {
	_, err := e.node.Call("put", ops.EncodePutArgs(key, value, ttl))
	return mapErr(err)
}

// PutBatch writes many key/value pairs, delegating to the node's shard-grouping
// bulk path so each shard takes one Raft log entry per chunk. Same durability as
// Put; ErrNotLeader propagates identically.
func (e *embedded) PutBatch(_ context.Context, entries []ops.PutEntry) error {
	return mapErr(e.node.PutBatch(entries))
}

// Del removes key through Raft. Returns (true, nil) if the entry existed,
// (false, nil) if absent, (false, err) on failure.
func (e *embedded) Del(_ context.Context, key []byte) (bool, error) {
	result, err := e.node.Call("del", ops.EncodeKeyArgs(key))
	if err != nil {
		return false, mapErr(err)
	}
	return len(result) > 0 && result[0] == 1, nil
}

// Call invokes a registered op by name with caller-encoded args.
func (e *embedded) Call(_ context.Context, op string, args []byte) ([]byte, error) {
	// Dispatch through the leader-following wrapper, not the bare node: a shard
	// this node hosts as a follower/backup would otherwise return NotLeader, which
	// mapErr flattens to ErrNotLeader — dropping the leader address and leaving
	// the caller nothing to act on. See the lf field for the full rationale.
	raw, err := e.lf.Call(op, args)
	if err != nil {
		return nil, mapErr(err)
	}
	return raw, nil
}

// callReadLeader dispatches a single-partition (P<=1) read op, routing to the
// shard's Raft leader when the read requested LeaderOnly/Linearizable
// consistency. For rc >= ConsistencyLeaderOnly it uses CallPhysical with
// leaderOnly=true so the read is pinned to the leader (CallPhysical re-routes on
// a NotLeader during serve), and — because the op args carry the rc byte — the
// leader's shard runs the readIndex barrier for a Linearizable read. For
// AnyReplica (rc==0) it keeps the plain Call path: load-balanced, no leader
// routing, no barrier — byte/behaviour-identical to the legacy single-shard read.
//
// The unpartitioned read previously used e.Call (= node.Call) for ALL
// consistency levels, which served on ANY hosting node (possibly a follower) and
// — even when rc was encoded — could not run the barrier because the read never
// reached the leader. This closes both holes (no leader routing AND barrier
// skipped) for P<=1 collections. CallPhysical resolves the target shard from the
// op's key extractor over args (the collection name embedded in args), so the
// logical collection name routes correctly for a P<=1 collection.
func (e *embedded) callReadLeader(op string, args []byte, rc uint8) ([]byte, error) {
	if rc < ops.ConsistencyLeaderOnly {
		return e.Call(context.Background(), op, args)
	}
	if rc == ops.ConsistencyBoundedStaleness {
		// BoundedStaleness deliberately routes ANY-REPLICA (NOT leader-pinned): the
		// serving shard enforces freshness via its bound guard and, when the replica
		// is too stale (or its leader-frontier RTT fails closed on a partition),
		// returns a *shard.NotLeaderError. Treat that as the "too-stale / upgrade"
		// signal and transparently RETRY once via the leader path (CallPhysical
		// leaderOnly=true) so a single-target bounded read still returns fresh data.
		// Use e.node.Call (NOT e.Call) so the too-stale signal arrives as a TYPED
		// *shard.NotLeaderError: e.Call applies mapErr, which collapses it into the
		// opaque ErrNotLeader sentinel that matchesNotLeader can no longer recognise —
		// the upgrade would then never fire and a too-stale bounded read would surface
		// not-leader to the client instead of transparently fetching fresh.
		raw, err := e.node.Call(op, args)
		if err == nil {
			return raw, nil
		}
		if matchesNotLeader(err) {
			raw, lerr := e.node.CallPhysical("", op, args, true)
			if lerr != nil {
				return nil, mapErr(lerr)
			}
			return raw, nil
		}
		return nil, mapErr(err)
	}
	raw, err := e.node.CallPhysical("", op, args, true)
	if err != nil {
		return nil, mapErr(err)
	}
	return raw, nil
}

// IsLeader reports whether this node is the current leader for key's shard.
func (e *embedded) IsLeader(key []byte) bool {
	return e.node.IsLocalLeader(key)
}

// LeaderAddr returns the current leader address for key's shard, or "" if
// unknown.
func (e *embedded) LeaderAddr(key []byte) string {
	return e.node.LeaderForKey(key)
}

// Close shuts down the embedded node and releases all resources.
func (e *embedded) Close() error {
	return e.node.Close()
}

func (e *embedded) VectorInsert(_ context.Context, collection string, id uint64, vec []float32, opts ...WriteOpts) error {
	collection = e.resolveAlias(collection)
	live, target, dual := e.dualTargets(collection, id)
	if live != "" {
		collection = live
	}
	wo := firstWriteOpts(opts)
	exp, hasExp := wo.expectedVersion()
	_, err := e.applyDualWrite("vector_insert", collection, target, dual, wo,
		func(phys string) []byte {
			return ops.EncodeVectorInsertArgsCAS(phys, id, vec, 0, nil, vector.SparseVector{}, exp, hasExp)
		})
	return err
}

// VectorSearch is the convenience form; use VectorSearchExt to observe FanMeta.
func (e *embedded) VectorSearch(_ context.Context, collection string, query []float32, k int) ([]VectorResult, error) {
	collection = e.resolveAlias(collection)
	if P, gen, ok := e.catalog.PartitionsGen(collection); ok && P > 1 {
		// Basic VectorSearch has no opts: default to AnyReplica (0) / Partial (0).
		// The cluster.FanResult (degraded flag / missing-partition list) is dropped
		// here — this is the convenience form; callers that need to observe FanMeta
		// use VectorSearchExt.
		res, _, err := e.denseFanOut(collection, P, gen, query, k, VectorFilter{}, 0, 0, 0)
		return res, err
	}
	body, err := e.Call(context.Background(), "vector_search", ops.EncodeVectorSearchArgs(collection, k, query))
	if err != nil {
		return nil, err
	}
	return ops.DecodeVectorSearchResults(body)
}

// Plan-1 fan-out scope (deferrals, so the limitations below are not mistaken for
// complete):
//   - In-process fan-out carries consistency via cluster.FanArgs (Consistency /
//     OnUnavailable) and reports degradation via cluster.FanResult. The wire
//     codecs in ops/vector.go — EncodeVectorSearchArgsOpts/DecodeVectorSearchArgsOpts
//     (consistency opts) and EncodeVectorSearchResultsDegraded/
//     DecodeVectorSearchResultsDegraded (degraded results) — are NOT yet wired
//     into this live path; they are plumbing for the future networked-coordinator
//     path.
//   - All vector ops now fan out across partitions (dense / filtered / MV /
//     hybrid / search_docs / groups / scroll / delete_by_filter), so none return
//     ErrPartitionedUnsupported anymore on a partitioned collection (P>1).
//   - Remote APIs (httpapi/grpcapi/client) are NOT partition-aware;
//     partitioned collections must be driven through the embedded backend.
//
// denseFanOut scatters a dense KNN search across all P physical partitions and
// merges the per-partition top-k by ascending distance. It honors ReadConsistency
// (AnyReplica vs LeaderOnly) and OnPartitionUnavailable (Partial vs Fail) via the
// fan-out coordinator. The returned FanResult flags whether any partition was
// skipped (Partial mode). Only used when P>1; the single-partition path is the
// untouched single-Call path in each caller.
func (e *embedded) denseFanOut(collection string, P int, gen uint32, query []float32, k int, filter VectorFilter, rc, opa uint8, bound uint64) ([]VectorResult, cluster.FanResult, error) {
	a := cluster.FanArgs{
		Collection:    collection,
		P:             P,
		Generation:    gen,
		K:             k,
		Op:            "vector_search",
		Consistency:   cluster.Consistency(rc),
		OnUnavailable: cluster.OnUnavailable(opa),
		Encode: func(physCol string) []byte {
			// The filter rides inside the per-partition args, so partition-local
			// filtering happens before the global top-k merge — exact for both the
			// unfiltered and filtered cases. The rc/opa trailer makes a Linearizable
			// read carry its consistency byte to the shard so the readIndex barrier
			// (ops.ReadConsistencyOf → verifyLeaderAndCatchUp) actually runs.
			return ops.EncodeVectorSearchArgsOpts(physCol, k, query, filter, rc, opa, bound)
		},
	}
	decode := func(raw []byte) ([]cluster.Scored, error) {
		results, err := ops.DecodeVectorSearchResults(raw)
		if err != nil {
			return nil, err
		}
		out := make([]cluster.Scored, len(results))
		for i, r := range results {
			out[i] = cluster.Scored{ID: r.ID, Dist: r.Distance, Score: r.Score}
		}
		return out, nil
	}
	// Ascending distance: nearest first (matches single-partition order).
	scored, fr, err := e.scatterMerge(a, decode, func(x, y cluster.Scored) bool { return x.Dist < y.Dist })
	if err != nil {
		return nil, fr, err
	}
	out := make([]VectorResult, len(scored))
	for i, s := range scored {
		out[i] = VectorResult{ID: s.ID, Distance: s.Dist, Score: s.Score}
	}
	return out, fr, nil
}

// scatterMerge is the shared fan-out core: it scatters a.Op to all a.P physical
// partitions via the node's physical caller, decodes each partition's payload
// into []cluster.Scored, and globally merges the top-a.K by the supplied less
// comparator. denseFanOut uses ascending distance. Keeping the boilerplate here
// means the only per-op differences are the Encode/decode/less closures.
// (mvFanOut deliberately does NOT use this path: it merges full
// vector.MultiResult values via the generic cluster.FanOut to preserve
// per-result Metadata, mirroring docsFanOut.)
func (e *embedded) scatterMerge(a cluster.FanArgs, decode func([]byte) ([]cluster.Scored, error), less func(x, y cluster.Scored) bool) ([]cluster.Scored, cluster.FanResult, error) {
	merge := func(parts [][]cluster.Scored, k int) []cluster.Scored {
		return cluster.MergeTopK(parts, k, less)
	}
	return cluster.FanOut(a, e.node.CallPhysical, decode, merge)
}

// mvFanOut scatters a multi-vector (late-interaction) search across all P
// physical partitions and merges the per-partition top-k by DESCENDING MaxSim
// score (higher relevance first), matching the single-partition ordering.
// Partition-local MaxSim rerank then a global top-k is exact because MaxSim is
// computed independently per document and IDs are disjoint across partitions.
// Only used when P>1; the single-partition path in VectorMVSearch is the
// untouched single-Call path.
//
// Like docsFanOut (and unlike the cluster.Scored-based scatterMerge), this
// merges FULL vector.MultiResult values via the generic cluster.FanOut, so each
// result's Metadata round-trips through fan-out exactly as it does on the
// single-shard path (EncodeMVResults/DecodeMVResults carry Metadata). Ties on
// Score are broken by ascending ID for deterministic ordering.
func (e *embedded) mvFanOut(name string, P int, gen uint32, query [][]float32, k, candPerToken int, filter VectorFilter, rc, opa uint8, bound uint64) ([]MultiResult, cluster.FanResult, error) {
	a := cluster.FanArgs{
		Collection:    name,
		P:             P,
		Generation:    gen,
		K:             k,
		Op:            "vector_mv_search",
		Consistency:   cluster.Consistency(rc),
		OnUnavailable: cluster.OnUnavailable(opa),
		Encode: func(physCol string) []byte {
			// The filter rides inside EVERY per-partition arg, so partition-local
			// MaxSim filtering happens before the global top-k merge — exact for
			// both the unfiltered and filtered cases (IDs are disjoint across
			// partitions; each shard returns its filtered top-k by MaxSim, the
			// coordinator unions + re-truncates). Mirrors denseFanOut. The rc/opa
			// trailer carries Linearizable to the shard so the readIndex barrier
			// runs on each partition's leader (vector_mv_search).
			return ops.EncodeMVSearchArgsOptsFilter(physCol, query, k, candPerToken, rc, opa, filter, bound)
		},
	}
	decode := func(raw []byte) ([]vector.MultiResult, error) {
		return ops.DecodeMVResults(raw)
	}
	merge := func(parts [][]vector.MultiResult, k int) []vector.MultiResult {
		var all []vector.MultiResult
		for _, p := range parts {
			all = append(all, p...)
		}
		// Descending MaxSim score (most relevant first); ascending ID on ties.
		sort.SliceStable(all, func(i, j int) bool {
			if all[i].Score != all[j].Score {
				return all[i].Score > all[j].Score
			}
			return all[i].ID < all[j].ID
		})
		if k >= 0 && len(all) > k {
			all = all[:k]
		}
		return all
	}
	return cluster.FanOut(a, e.node.CallPhysical, decode, merge)
}

// mvScrollFanOut scatters vector_mv_scroll to all partitions, unions, GLOBAL-sorts
// by ascending id, truncates to limit, and derives next_cursor — the MV mirror of
// scrollFanOut / namedScrollFanOut (the merge is id-ascending, NOT the score merge
// in mvFanOut). Each partition receives the SAME global (filter, limit, afterID);
// rc/opa ride EVERY per-partition arg so each shard's data barrier fires
// (anti-silent-drop). MV doc ids are disjoint across partitions, so the truncated
// global-sorted union is the correct next page — gap-free + dup-free. The next
// page sends afterID = next_cursor's id to ALL partitions again.
func (e *embedded) mvScrollFanOut(name string, P int, gen uint32, filter VectorFilter, limit int, rc, opa uint8, afterID uint64, hasAfter bool, order *ops.ScrollOrder, bound uint64) ([]VectorDocument, cluster.FanResult, string, error) {
	a := cluster.FanArgs{
		Collection:    name,
		P:             P,
		Generation:    gen,
		K:             limit,
		Op:            "vector_mv_scroll",
		Consistency:   cluster.Consistency(rc),
		OnUnavailable: cluster.OnUnavailable(opa),
		Encode: func(physCol string) []byte {
			// rc + the SAME global cursor/order ride every per-partition arg so each
			// shard's data barrier fires and every shard pages from the same bound.
			return ops.EncodeMVScrollArgsOrderBounded(physCol, filter, limit, rc, opa, afterID, hasAfter, order, bound)
		},
	}
	decode := func(raw []byte) ([]vector.Document, error) {
		return ops.DecodeVectorDocs(raw)
	}
	merge := func(parts [][]vector.Document, _ int) []vector.Document {
		var all []vector.Document
		for _, p := range parts {
			all = append(all, p...)
		}
		return all
	}
	all, fr, err := cluster.FanOut(a, e.node.CallPhysical, decode, merge)
	if err != nil {
		return nil, fr, "", err
	}
	if order != nil {
		// GLOBAL (value, id) merge: sort the union by the order's total order (NOT id),
		// then truncate. Disjoint per-partition ids ⇒ no equal-comparing dups; missing-field
		// points were already EXCLUDED per partition. See scrollFanOut for the re-derivation.
		ob := scrollOrderByFromOps(order)
		all = sortDocsByOrder(all, ob)
		if limit > 0 && len(all) > limit {
			all = all[:limit]
		}
		return all, fr, scrollNextCursorOrder(all, limit, ob), nil
	}
	// GLOBAL id-ascending merge of the per-partition pages. IDs are disjoint across
	// partitions so the sort only orders the union, not equal-ID duplicates.
	sort.SliceStable(all, func(i, j int) bool { return all[i].ID < all[j].ID })
	if limit > 0 && len(all) > limit {
		all = all[:limit]
	}
	return all, fr, scrollNextCursor(all, limit), nil
}

// VectorSearchInto decodes the (read-only, Raft-bypassing) search response
// straight into dst, saving the result-slice allocation when dst is reused.
// Convenience form; use VectorSearchExt to observe FanMeta.
func (e *embedded) VectorSearchInto(_ context.Context, collection string, query []float32, k int, dst []VectorResult) ([]VectorResult, error) {
	collection = e.resolveAlias(collection)
	if P, gen, ok := e.catalog.PartitionsGen(collection); ok && P > 1 {
		// Fan-out does not reuse dst (results are merged across partitions); return
		// a freshly merged slice. The dst optimization applies to the single-Call
		// path only. FanResult dropped — convenience form; see VectorSearchExt.
		res, _, err := e.denseFanOut(collection, P, gen, query, k, VectorFilter{}, 0, 0, 0)
		return res, err
	}
	body, err := e.Call(context.Background(), "vector_search", ops.EncodeVectorSearchArgs(collection, k, query))
	if err != nil {
		return nil, err
	}
	return ops.DecodeVectorSearchResultsInto(body, dst)
}

func (e *embedded) VectorDelete(_ context.Context, collection string, id uint64, opts ...WriteOpts) (bool, error) {
	collection = e.resolveAlias(collection)
	live, target, dual := e.dualTargets(collection, id)
	if live != "" {
		collection = live
	}
	// Dual-delete removes from both gens; the returned existed-flag reflects the
	// live (read source-of-truth) gen.
	wo := firstWriteOpts(opts)
	exp, hasExp := wo.expectedVersion()
	body, err := e.applyDualWrite("vector_delete", collection, target, dual, wo,
		func(phys string) []byte { return ops.EncodeVectorDeleteArgsCAS(phys, id, exp, hasExp) })
	if err != nil {
		return false, err
	}
	return len(body) == 1 && body[0] == 1, nil
}

// VectorInsertIfAbsent routes to the live gen's physical partition (same as
// VectorInsert) and runs the atomic insert-if-absent op there. It is
// intentionally NOT dual-written during a reshard: it is the copy-loop
// primitive that populates the NEW gen (reshard Phase 3), so the orchestrator
// targets the new-gen partition explicitly. User-facing point writes
// (VectorInsert/Upsert/Delete) dual-write via dualTargets; this stays
// single-target by design.
func (e *embedded) VectorInsertIfAbsent(_ context.Context, collection string, id uint64, vec []float32, opts VectorInsertOpts) (bool, error) {
	collection = e.resolveAlias(collection)
	if phys, ok := e.partitionOf(collection, id); ok {
		collection = phys
	}
	body, err := e.Call(context.Background(), "vector_insert_if_absent", ops.EncodeVectorInsertArgsExt(collection, id, vec, opts.TTL, opts.Metadata, opts.Sparse))
	if err != nil {
		return false, err
	}
	return ops.DecodeIfAbsentResult(body)
}

// VectorExists probes the live gen's physical partition for id's liveness.
func (e *embedded) VectorExists(_ context.Context, collection string, id uint64) (bool, error) {
	collection = e.resolveAlias(collection)
	if phys, ok := e.partitionOf(collection, id); ok {
		collection = phys
	}
	body, err := e.Call(context.Background(), "vector_exists", ops.EncodeExistsArgs(collection, id))
	if err != nil {
		return false, err
	}
	return ops.DecodeExistsResult(body)
}

// VectorGet retrieves a dense point by id from the live gen's owning physical
// partition (route-by-id, like VectorExists). not-found is the decoded found=0 FLAG
// (never an error). Partitioned fan-out is intercepted earlier; this routes to the
// single owning partition, which is correct because a point lives in exactly one.
// VectorGet is the back-compat convenience form (AnyReplica): it delegates to
// VectorGetExt with a zero ReadOpts, so its wire + behaviour are byte-identical to
// the legacy single-Call path.
func (e *embedded) VectorGet(ctx context.Context, collection string, id uint64, withVector, withPayload bool) (bool, []float32, VectorMetadata, time.Duration, *VectorSparse, error) {
	return e.VectorGetExt(ctx, collection, id, withVector, withPayload, ReadOpts{})
}

// VectorGetExt retrieves a dense point by id with read-consistency. A point lives
// in exactly ONE partition (route-by-id), so a Linearizable get routes to THAT
// partition's leader and arms the shard readIndex barrier (read-your-writes) —
// no fan-out. For rc==0 (AnyReplica) it is the legacy plain-Call path, wire- and
// behaviour-identical to the old VectorGet.
func (e *embedded) VectorGetExt(_ context.Context, collection string, id uint64, withVector, withPayload bool, opts ReadOpts) (bool, []float32, VectorMetadata, time.Duration, *VectorSparse, error) {
	collection, err := e.resolveCollectionForRead(collection, opts.ReadConsistency, time.Now().Add(metaReadIndexReadTimeout))
	if err != nil {
		return false, nil, nil, 0, nil, err
	}
	if phys, ok := e.partitionOf(collection, id); ok {
		collection = phys
	}
	body, err := e.callReadLeader("vector_get",
		ops.EncodeVectorGetArgsOpts(collection, id, getFlags(withVector, withPayload), opts.ReadConsistency, opts.OnPartitionUnavailable, opts.MaxStaleness),
		opts.ReadConsistency)
	if err != nil {
		return false, nil, nil, 0, nil, err
	}
	return ops.DecodeVectorGetResult(body)
}

// VectorGetBatch retrieves many dense points by id in one op. For an
// unpartitioned (or P<=1) collection it is a SINGLE vector_get_batch Call with
// all (deduped) ids. For a partitioned collection it groups the ids by their
// owning partition and asks each partition ONLY for its owned subset
// concurrently (getBatchScatter), then merges. A partial miss is normal (the
// absent ids land in `missing`, never an error); points + missing are returned
// sorted ascending by id. Like VectorGet this is AnyReplica (no read
// consistency) and route-by-id correct: a point lives in exactly one partition.
func (e *embedded) VectorGetBatch(_ context.Context, collection string, ids []uint64, withVector, withPayload bool) ([]BatchGetPoint, []uint64, error) {
	collection = e.resolveAlias(collection)
	flags := getFlags(withVector, withPayload)
	// Dedup up front: a repeated id is fetched once and appears once in the
	// result. Preserves first-seen membership; final order is id-sorted anyway.
	ids = dedupIDs(ids)
	if len(ids) == 0 {
		return nil, nil, nil
	}

	P, gen, ok := e.catalog.PartitionsGen(collection)
	if !ok || P <= 1 {
		// Unpartitioned / single-partition fast path: ONE call with all ids,
		// mirroring how VectorGet uses the logical name on the unpartitioned path.
		body, err := e.Call(context.Background(), "vector_get_batch", ops.EncodeVectorGetBatchArgs(collection, ids, flags))
		if err != nil {
			return nil, nil, err
		}
		// Decode into per-call scratch: the row slice is sized from the REQUEST
		// (one row per requested id) and every vector is carved out of one arena
		// instead of a make([]float32, dim) per row. The scratch is fresh per call
		// — the decoded vectors are handed to our caller inside the returned
		// points, so it can never be pooled across requests.
		rows, _, err := ops.DecodeVectorGetBatchResultInto(body, make([]ops.GetBatchRow, 0, len(ids)), nil)
		if err != nil {
			return nil, nil, err
		}
		return splitBatchRows(rows)
	}
	return e.getBatchScatter(collection, P, gen, ids, flags)
}

// getBatchScatter groups ids by their owning partition (ops.PartitionOf) and
// queries each non-empty partition CONCURRENTLY for ONLY its owned subset, then
// merges the per-partition rows into the global points + missing. Concurrency is
// bounded by the number of partitions touched (one goroutine per non-empty
// group, mirroring cluster.FanOut / mvFanOut's one-goroutine-per-partition
// model — P is a small fixed count). Each goroutine writes only its own result
// slot, so there is no shared-state race; the slots are combined after the wait.
// An unreachable partition fails the WHOLE batch (fail-loud, the strictest read
// default) — a partition error is NEVER silently converted into "missing", so a
// shard outage cannot masquerade as deleted data. points + missing are sorted
// ascending by id.
func (e *embedded) getBatchScatter(collection string, P int, gen uint32, ids []uint64, flags uint8) ([]BatchGetPoint, []uint64, error) {
	all, err := scatterIDsByPartition(e, collection, P, gen, ids, func(phys string, sub []uint64) ([]ops.GetBatchRow, error) {
		raw, err := e.node.CallPhysical(phys, "vector_get_batch", ops.EncodeVectorGetBatchArgs(phys, sub, flags), false)
		if err != nil {
			return nil, err
		}
		// Per-partition scratch, allocated inside the fetch: these fetches run
		// CONCURRENTLY (one goroutine per partition) and the decoded vectors alias
		// the arena, so the scratch must not be shared between partitions — nor
		// reused across requests, since the vectors escape into the merged result.
		rows, _, err := ops.DecodeVectorGetBatchResultInto(raw, make([]ops.GetBatchRow, 0, len(sub)), nil)
		return rows, err
	})
	if err != nil {
		return nil, nil, err
	}
	return splitBatchRows(all)
}

// scatterIDsByPartition is the family-agnostic scatter core shared by dense /
// named / MV batch get. It groups ids by their owning partition
// (ops.PartitionOf), queries each non-empty partition CONCURRENTLY for ONLY its
// owned subset via fetch (one goroutine per non-empty group — P is a small fixed
// count, mirroring cluster.FanOut's one-goroutine-per-partition model), and
// concatenates the per-partition rows. Each goroutine writes only its own result
// slot, so there is no shared-state race; the slots are combined after the wait.
// An unreachable partition fails the WHOLE batch (fail-loud, the strictest read
// default) — a partition error is NEVER silently converted into "missing", so a
// shard outage cannot masquerade as deleted data. The physical collection name
// passed to fetch is string(ops.PartitionKeyGen(collection, gen, p)). R is the
// family's row type; callers split the returned rows into present/missing.
func scatterIDsByPartition[R any](e *embedded, collection string, P int, gen uint32, ids []uint64, fetch func(phys string, sub []uint64) ([]R, error)) ([]R, error) {
	// P<=0 is not a valid partition count (there is nothing to scatter to);
	// guard it explicitly rather than let ops.PartitionOf(id, P) — which
	// returns 0 for P<=1 — index the P-length counts/groups slices below with
	// a 0 index into a zero-length slice.
	if P <= 0 {
		return nil, nil
	}
	// P is a small fixed count, so a slice indexed by partition replaces the
	// map[int][]uint64 this used to build (map insert/lookup overhead per id,
	// for no benefit over direct indexing). A counting pass sizes each
	// sub-slice exactly, so the append loop below never grows one.
	counts := make([]int, P)
	for _, id := range ids {
		counts[ops.PartitionOf(id, P)]++
	}
	groups := make([][]uint64, P)
	for p, n := range counts {
		if n > 0 {
			groups[p] = make([]uint64, 0, n)
		}
	}
	for _, id := range ids {
		p := ops.PartitionOf(id, P)
		groups[p] = append(groups[p], id)
	}

	type result struct {
		rows []R
		err  error
	}
	parts := make([]int, 0, P)
	for p := 0; p < P; p++ {
		if len(groups[p]) > 0 {
			parts = append(parts, p)
		}
	}
	results := make([]result, len(parts))
	runPart := func(i, p int) {
		phys := string(ops.PartitionKeyGen(collection, gen, p))
		sub := groups[p]
		rows, err := fetch(phys, sub)
		results[i] = result{rows: rows, err: err}
	}
	var wg sync.WaitGroup
	// The last non-empty partition runs on the CALLING goroutine (same
	// rationale as cluster.FanOut: the caller only wg.Wait()s afterward), the
	// rest are spawned as before.
	if n := len(parts); n > 0 {
		for i := 0; i < n-1; i++ {
			wg.Add(1)
			go func(i, p int) {
				defer wg.Done()
				runPart(i, p)
			}(i, parts[i])
		}
		runPart(n-1, parts[n-1])
	}
	wg.Wait()

	var all []R
	for i, r := range results {
		if r.err != nil {
			// Fail-loud: an unreachable / erroring partition fails the batch rather
			// than dropping its ids into missing (which would look like deletes).
			return nil, fmt.Errorf("partition %d: %w", parts[i], r.err)
		}
		all = append(all, r.rows...)
	}
	return all, nil
}

// dedupIDs returns ids with duplicates removed, preserving first-seen order. A
// nil/empty input yields a nil slice. The output order is not load-bearing
// (VectorGetBatch id-sorts the final result), but first-seen order keeps the
// per-partition subsets stable for tests that observe the encoded args.
func dedupIDs(ids []uint64) []uint64 {
	if len(ids) == 0 {
		return nil
	}
	seen := make(map[uint64]struct{}, len(ids))
	out := make([]uint64, 0, len(ids))
	for _, id := range ids {
		if _, ok := seen[id]; ok {
			continue
		}
		seen[id] = struct{}{}
		out = append(out, id)
	}
	return out
}

// splitBatchRows partitions decoded vector_get_batch rows into the present
// points (Found) and the missing ids (!Found), then sorts both ascending by id
// for a deterministic result. Each present point carries its id plus the
// projected vec/meta/ttl/sparse the handler returned.
func splitBatchRows(rows []ops.GetBatchRow) ([]BatchGetPoint, []uint64, error) {
	// points is preallocated to len(rows) (the worst case, all found) so the
	// append loop below never grows it. missing is intentionally left nil
	// here rather than preallocated: an all-found batch — the common case —
	// then costs it nothing, and callers that check `missing == nil` for "no
	// misses" keep working (preallocating both would turn that into a
	// non-nil empty slice and waste the up-to-len(rows) backing array on a
	// batch with zero misses).
	points := make([]BatchGetPoint, 0, len(rows))
	var missing []uint64
	for _, r := range rows {
		if r.Found {
			points = append(points, BatchGetPoint{
				ID:      r.ID,
				Vec:     r.Vec,
				Meta:    r.Meta,
				TTL:     time.Duration(r.TTLMs) * time.Millisecond,
				Sparse:  r.Sparse,
				Version: r.Version,
			})
		} else {
			missing = append(missing, r.ID)
		}
	}
	slices.SortFunc(points, func(a, b BatchGetPoint) int { return cmp.Compare(a.ID, b.ID) })
	slices.Sort(missing)
	return points, missing, nil
}

// splitNamedBatchRows partitions decoded vector_named_get_batch rows into the
// present points (Found) and the missing ids (!Found), then sorts both ascending
// by id for a deterministic result. Each present point carries its id plus the
// projected vectors-map / meta / ttl the handler returned. The named clone of
// splitBatchRows.
func splitNamedBatchRows(rows []ops.NamedGetBatchRow) ([]NamedBatchGetPoint, []uint64, error) {
	// See splitBatchRows: points is preallocated, missing stays nil until the
	// first miss so an all-found batch doesn't turn `missing == nil` into a
	// non-nil empty slice for callers that check it, and doesn't pay for a
	// backing array it never uses.
	points := make([]NamedBatchGetPoint, 0, len(rows))
	var missing []uint64
	for _, r := range rows {
		if r.Found {
			points = append(points, NamedBatchGetPoint{
				ID:      r.ID,
				Vectors: r.Vectors,
				Meta:    r.Meta,
				TTL:     time.Duration(r.TTLMs) * time.Millisecond,
				Version: r.Version,
			})
		} else {
			missing = append(missing, r.ID)
		}
	}
	slices.SortFunc(points, func(a, b NamedBatchGetPoint) int { return cmp.Compare(a.ID, b.ID) })
	slices.Sort(missing)
	return points, missing, nil
}

// splitMVBatchRows partitions decoded vector_mv_get_batch rows into the present
// points (Found) and the missing ids (!Found), then sorts both ascending by id
// for a deterministic result. Each present point carries its id plus the
// projected token matrix / meta the handler returned (MV has NO ttl). The MV
// clone of splitNamedBatchRows.
func splitMVBatchRows(rows []ops.MVGetBatchRow) ([]MVBatchGetPoint, []uint64, error) {
	var points []MVBatchGetPoint
	var missing []uint64
	for _, r := range rows {
		if r.Found {
			points = append(points, MVBatchGetPoint{
				ID:      r.ID,
				Tokens:  r.Tokens,
				Meta:    r.Meta,
				Version: r.Version,
			})
		} else {
			missing = append(missing, r.ID)
		}
	}
	sort.Slice(points, func(i, j int) bool { return points[i].ID < points[j].ID })
	sort.Slice(missing, func(i, j int) bool { return missing[i] < missing[j] })
	return points, missing, nil
}

// VectorSetPayload merges patch into the point's payload on the live gen's owning
// physical partition. applied=0 (the decoded flag) for an absent point — not an
// error. Like VectorDelete it dual-writes during a reshard (via dualTargets /
// applyDualWrite) so the new gen is not stale after cutover; the applied-flag
// reflects the live (read source-of-truth) gen. A set/merge against an absent id
// on the target gen is a harmless no-op (applied=0 there), so dual-write is safe.
func (e *embedded) VectorSetPayload(_ context.Context, collection string, id uint64, patch VectorMetadata, keyTTLMs map[string]int64, opts ...WriteOpts) (bool, error) {
	collection = e.resolveAlias(collection)
	live, target, dual := e.dualTargets(collection, id)
	if live != "" {
		collection = live
	}
	wo := firstWriteOpts(opts)
	exp, hasExp := wo.expectedVersion()
	body, err := e.applyDualWrite("vector_set_payload", collection, target, dual, wo,
		func(phys string) []byte { return ops.EncodeSetPayloadArgsCAS(phys, id, patch, keyTTLMs, exp, hasExp) })
	if err != nil {
		return false, err
	}
	return ops.DecodePayloadResult(body)
}

// VectorOverwritePayload replaces the point's payload on the owning partition,
// dual-writing during a reshard (mirror VectorSetPayload / VectorDelete).
func (e *embedded) VectorOverwritePayload(_ context.Context, collection string, id uint64, meta VectorMetadata, keyTTLMs map[string]int64, opts ...WriteOpts) (bool, error) {
	collection = e.resolveAlias(collection)
	live, target, dual := e.dualTargets(collection, id)
	if live != "" {
		collection = live
	}
	wo := firstWriteOpts(opts)
	exp, hasExp := wo.expectedVersion()
	body, err := e.applyDualWrite("vector_overwrite_payload", collection, target, dual, wo,
		func(phys string) []byte { return ops.EncodeSetPayloadArgsCAS(phys, id, meta, keyTTLMs, exp, hasExp) })
	if err != nil {
		return false, err
	}
	return ops.DecodePayloadResult(body)
}

// VectorDeletePayloadKeys removes the listed payload keys on the owning partition,
// dual-writing during a reshard (mirror VectorSetPayload / VectorDelete).
func (e *embedded) VectorDeletePayloadKeys(_ context.Context, collection string, id uint64, keys []string, opts ...WriteOpts) (bool, error) {
	collection = e.resolveAlias(collection)
	live, target, dual := e.dualTargets(collection, id)
	if live != "" {
		collection = live
	}
	wo := firstWriteOpts(opts)
	exp, hasExp := wo.expectedVersion()
	body, err := e.applyDualWrite("vector_delete_payload_keys", collection, target, dual, wo,
		func(phys string) []byte { return ops.EncodeDeletePayloadKeysArgsCAS(phys, id, keys, exp, hasExp) })
	if err != nil {
		return false, err
	}
	return ops.DecodePayloadResult(body)
}

// VectorClearPayload empties the point's payload on the owning partition,
// dual-writing during a reshard (mirror VectorSetPayload / VectorDelete).
func (e *embedded) VectorClearPayload(_ context.Context, collection string, id uint64, opts ...WriteOpts) (bool, error) {
	collection = e.resolveAlias(collection)
	live, target, dual := e.dualTargets(collection, id)
	if live != "" {
		collection = live
	}
	wo := firstWriteOpts(opts)
	exp, hasExp := wo.expectedVersion()
	body, err := e.applyDualWrite("vector_clear_payload", collection, target, dual, wo,
		func(phys string) []byte { return ops.EncodeClearPayloadArgsCAS(phys, id, exp, hasExp) })
	if err != nil {
		return false, err
	}
	return ops.DecodePayloadResult(body)
}

func (e *embedded) CreateCollection(_ context.Context, name string, cfg VectorConfig) error {
	if strings.ContainsAny(name, "#@") {
		return fmt.Errorf("vector: collection name %q must not contain reserved characters '#' or '@'", name)
	}
	// Alias-shadow guard: a new collection must not take the name of an existing
	// alias, or data-plane ops would be ambiguous (alias resolution vs the real
	// collection). Mirror the PartitionsGen re-partition guard placement below.
	if _, ok := e.catalog.ResolveAlias(name); ok {
		return fmt.Errorf("vector: collection name %q is already an alias: %w", name, ErrAliasShadowsCollection)
	}
	// Cross-type / re-partition guard (fail-loud, all Partitions values): the
	// catalog is shared with the MV family — a name already partitioned as
	// either type must never be re-partitioned. NOTE: resplit re-creates
	// physical partitions via e.Call directly (not through CreateCollection), so
	// it is unaffected by this guard.
	if _, _, ok := e.catalog.PartitionsGen(name); ok {
		return fmt.Errorf("vector: collection %q is already partitioned", name)
	}
	// Partitioned dense: reject if a multi-vector collection of the same name
	// exists (err==nil means the MV probe found it).
	if cfg.Partitions > 1 {
		if _, err := e.Call(context.Background(), "vector_mv_get_config", ops.EncodeMVGetConfigArgs(name)); err == nil {
			return fmt.Errorf("vector: a multi-vector collection named %q already exists", name)
		}
	}
	// Single-partition path: byte-for-byte unchanged (no catalog write, no extra
	// collections).
	if cfg.Partitions <= 1 {
		_, err := e.Call(context.Background(), "vector_create_collection", ops.EncodeCreateCollectionArgs(name, cfg))
		return err
	}

	// Partitioned path: create the logical collection (so the logical name still
	// resolves), then one physical single-partition collection per partition, and
	// record P in the catalog. Physical collections are themselves single-
	// partition (Partitions reset to 0) so they never recurse.
	P := cfg.Partitions
	// The logical collection is a single-partition existence/resolution marker;
	// the catalog (SetPartitionsGen below) is the sole source of truth for P.
	// Creating it with Partitions=0 keeps a forwarded logical-create from being
	// re-expanded by a remote node's fanout dispatcher (which only expands
	// Partitions>1), which would otherwise race this coordinator's physical-create
	// loop into "already exists".
	physCfg := cfg
	physCfg.Partitions = 0
	if _, err := e.Call(context.Background(), "vector_create_collection", ops.EncodeCreateCollectionArgs(name, physCfg)); err != nil {
		return err
	}
	for p := 0; p < P; p++ {
		phys := string(ops.PartitionKeyGen(name, 0, p))
		_, err := e.Call(context.Background(), "vector_create_collection", ops.EncodeCreateCollectionArgs(phys, physCfg))
		if err != nil {
			e.rollbackPartialPartitionedCreate(context.Background(), name, p, "vector_drop_collection", ops.EncodeDropCollectionArgs)
			return fmt.Errorf("create partition %d/%d for %q: %w", p, P, name, err)
		}
	}
	if err := e.catalog.SetPartitionsGen(name, P, 0); err != nil {
		// The catalog entry is what MAKES the collection partitioned. Without it
		// the P physical collections above are unreachable garbage and the logical
		// name resolves to an empty single-partition collection — the same broken
		// half-state as a failure inside the loop, so it unwinds the same way.
		e.rollbackPartialPartitionedCreate(context.Background(), name, P, "vector_drop_collection", ops.EncodeDropCollectionArgs)
		return err
	}
	return nil
}

// rollbackPartialPartitionedCreate unwinds a partitioned create that failed
// part-way: it drops the `made` physical partitions already created, then the
// logical marker collection.
//
// WHY IT HAS TO EXIST. A partitioned create is P+1 separate cluster ops with no
// transaction around them. Before this, a failure at partition k returned the
// error and walked away, leaving the logical collection plus partitions 0..k-1
// behind and NO catalog entry. That state is worse than either outcome the
// caller can reason about: the collection is not usable (nothing records that it
// is partitioned, so reads and writes resolve to the empty logical marker), and
// it is not absent either — a retry of the identical create fails with "already
// exists", so the caller cannot get out of it by repeating the request. Dropping
// it is awkward for the same reason: the drop path consults the catalog to find
// the partitions, and the catalog has no entry.
//
// Rollback is deliberately BEST-EFFORT and errors are discarded. It runs while
// returning another error — the create's own, which is the one the caller needs
// — and a drop failure here means the cluster is already in trouble in a way a
// second error string would not help with. Dropping is idempotent, so a partial
// rollback that is retried later still converges. Partitions unwind in reverse
// order purely so the observable sequence mirrors the creates.
func (e *embedded) rollbackPartialPartitionedCreate(ctx context.Context, name string, made int, dropOp string, encode func(string) []byte) {
	for p := made - 1; p >= 0; p-- {
		phys := string(ops.PartitionKeyGen(name, 0, p))
		_, _ = e.Call(ctx, dropOp, encode(phys))
	}
	_, _ = e.Call(ctx, dropOp, encode(name))
}

// Alias-management sentinel errors. These surface from the alias coordinator ops
// (AliasBatch/CreateAlias) and mirror the existing vector error sentinels so the
// transports can map them to 400 / InvalidArgument.
//
// IMPORTANT: the "rostam: alias " prefix shared by all four messages is
// load-bearing. httpapi.statusForError and grpcapi.grpcError match this prefix
// via strings.Contains to route alias validation errors to 400 / InvalidArgument
// (the sentinels live in the root package and cannot be imported by the
// transports without an import cycle). Do NOT change the prefix without updating
// both transport matchers — a reworded message silently falls through to
// 500 / Internal.
var (
	// ErrAliasTargetMissing: a create action's target collection does not exist.
	ErrAliasTargetMissing = errors.New("rostam: alias target collection does not exist")
	// ErrAliasShadowsCollection: an alias name collides with an existing real
	// collection (or, on create-collection, the new name collides with an alias).
	ErrAliasShadowsCollection = errors.New("rostam: alias name shadows an existing collection")
	// ErrAliasTargetIsAlias: a create action's target is itself an alias (only one
	// level of indirection is allowed — targets must be real collections).
	ErrAliasTargetIsAlias = errors.New("rostam: alias target is itself an alias")
	// ErrAliasReservedChar: an alias name contains a reserved '#'/'@' character
	// (those are reserved for physical partition / generation names).
	ErrAliasReservedChar = errors.New("rostam: alias name must not contain reserved characters '#' or '@'")
)

// collectionExists reports whether a real (non-alias) collection of the given
// name exists in ANY of the three families (dense/named/MV) OR has a partition
// catalog entry. A collection exists if it is partitioned (catalog entry) or a
// config probe in any family succeeds. Physical/internal '#'/'@' names are not
// valid alias targets, but the caller guards alias names; targets are user
// collection names so a probe is the source of truth.
func (e *embedded) collectionExists(ctx context.Context, name string) bool {
	if _, _, ok := e.catalog.PartitionsGen(name); ok {
		return true
	}
	if _, err := e.Call(ctx, "vector_get_config", ops.EncodeGetConfigArgs(name)); err == nil {
		return true
	}
	if _, err := e.Call(ctx, "vector_mv_get_config", ops.EncodeMVGetConfigArgs(name)); err == nil {
		return true
	}
	if _, err := e.Call(ctx, "vector_named_get_config", ops.EncodeNamedNameArgs(name)); err == nil {
		return true
	}
	return false
}

// AliasBatch atomically applies a batch of alias mutations (the alias_batch
// coordinator op). It is a meta-Raft metadata op, NOT shard-routed — it mirrors
// the VectorReshard coordinator pattern. The WHOLE batch is validated BEFORE
// commit: if any create action is invalid the entire batch is rejected and
// nothing is applied. Within a batch a create is an upsert (overwrite), so an
// atomic swap {delete prod, create prod→v2} is valid. Validation per create:
//   - the target collection must EXIST (real, any family);
//   - the alias name must NOT shadow an existing real collection;
//   - the alias name must have no '#'/'@' reserved chars;
//   - the target must NOT itself be an alias (one level only).
//
// Delete actions are always valid (absent alias → no-op delete).
func (e *embedded) AliasBatch(ctx context.Context, actions []AliasAction) error {
	for _, a := range actions {
		if a.Delete {
			continue
		}
		if strings.ContainsAny(a.Alias, "#@") {
			return fmt.Errorf("alias %q: %w", a.Alias, ErrAliasReservedChar)
		}
		// The alias name must not shadow a REAL collection. (It may overwrite an
		// existing alias — that is an upsert/swap and is allowed.)
		if e.collectionExists(ctx, a.Alias) {
			return fmt.Errorf("alias %q: %w", a.Alias, ErrAliasShadowsCollection)
		}
		// The target must not itself be an alias (one level of indirection only).
		// Checked BEFORE the existence probe so target==alias surfaces the precise
		// ErrAliasTargetIsAlias rather than ErrAliasTargetMissing (an alias name is
		// not a real collection, so the existence probe would also fail).
		if _, ok := e.catalog.ResolveAlias(a.Canonical); ok {
			return fmt.Errorf("alias %q → %q: %w", a.Alias, a.Canonical, ErrAliasTargetIsAlias)
		}
		// The target must exist as a real collection.
		if !e.collectionExists(ctx, a.Canonical) {
			return fmt.Errorf("alias %q → %q: %w", a.Alias, a.Canonical, ErrAliasTargetMissing)
		}
	}
	return e.catalog.SetAliases(actions)
}

// CreateAlias creates (or overwrites — upsert) an alias pointing at a real
// collection. It builds a one-action batch and delegates to AliasBatch so the
// validation + commit path is shared.
func (e *embedded) CreateAlias(ctx context.Context, alias, collection string) error {
	return e.AliasBatch(ctx, []AliasAction{{Alias: alias, Canonical: collection}})
}

// DeleteAlias removes an alias (the alias_batch coordinator op with a single
// delete action). An absent alias is a no-op.
func (e *embedded) DeleteAlias(ctx context.Context, alias string) error {
	return e.AliasBatch(ctx, []AliasAction{{Alias: alias, Delete: true}})
}

// aliasDisplayName converts a canonical catalog name back to the user-facing
// form by stripping the implicit "default/" namespace prefix; names in an
// explicit namespace are returned unchanged. This makes the list API mirror the
// bare names the user passed to CreateAlias (the catalog stores canonical→
// canonical for routing; resolution canonicalizes on lookup).
func aliasDisplayName(canonical string) string {
	return strings.TrimPrefix(canonical, "default/")
}

// ListAliases returns the alias→collection map (the alias_list coordinator op, a
// local FSM read — no Raft). Names are returned in user-facing display form
// (implicit "default/" namespace stripped). When collection != "" the result is
// filtered to aliases whose target == collection (compared canonically).
func (e *embedded) ListAliases(_ context.Context, collection string) (map[string]string, error) {
	all := e.catalog.ListAliases()
	var want string
	if collection != "" {
		want = ops.CanonicalName(collection)
	}
	out := make(map[string]string, len(all))
	for alias, target := range all {
		if want != "" && target != want {
			continue
		}
		out[aliasDisplayName(alias)] = aliasDisplayName(target)
	}
	return out, nil
}

// fanoutDrop is the shared drop-fanout for both the dense and multi-vector
// families. It drops every physical partition of the live generation [0,P)
// using the given drop op (encode produces the per-physical args), then the
// empty logical collection, then neutralizes the catalog entry to P=1
// (PartitionsGen treats P<=1 as unpartitioned). Orphan generations left by a
// failed resplit are intentionally not touched here — the live-gen-only
// invariant leaves them to VectorResplitCleanup.
func (e *embedded) fanoutDrop(ctx context.Context, collection string, P int, gen uint32, dropOp string, encode func(phys string) []byte) error {
	for p := 0; p < P; p++ {
		phys := string(ops.PartitionKeyGen(collection, gen, p))
		if _, err := e.Call(ctx, dropOp, encode(phys)); err != nil {
			return fmt.Errorf("drop partition %d/%d for %q: %w", p, P, collection, err)
		}
	}
	if _, err := e.Call(ctx, dropOp, encode(collection)); err != nil {
		return fmt.Errorf("drop logical collection %q: %w", collection, err)
	}
	return e.catalog.SetPartitionsGen(collection, 1, 0)
}

// dropCollectionFanout drops a partitioned dense collection. See fanoutDrop.
func (e *embedded) dropCollectionFanout(ctx context.Context, collection string, P int, gen uint32) error {
	return e.fanoutDrop(ctx, collection, P, gen, "vector_drop_collection", func(phys string) []byte {
		return ops.EncodeDropCollectionArgs(phys)
	})
}

// resolveAlias maps an alias name to its canonical target collection (the
// write-through data-plane chokepoint). It is called at the TOP of every embedded
// data-plane method (before any partition routing) AND in
// fanoutDispatcher.partitioned() — both MUST resolve to the SAME canonical name
// (LANDMINE #1: a mismatch makes a partitioned collection searched via an alias
// hit the empty logical collection → silent zero results).
//
// Short-circuit: a name containing a reserved '#'/'@' char is a physical
// partition / generation name (e.g. "docs#2", "docs@1#2") built by
// PartitionKeyGen, NEVER an alias — return it unchanged so reshard / fan-out
// physical-name ops are never rerouted. One level only: the catalog stores
// canonical→canonical and ResolveAlias canonicalizes the input, so a single
// lookup is exact (targets are real collections, never aliases). Zero-cost when
// the name is not an alias (one local catalog map read, no Raft).
func (e *embedded) resolveAlias(name string) string {
	if strings.ContainsAny(name, "#@") {
		return name
	}
	if canon, ok := e.catalog.ResolveAlias(name); ok {
		return canon
	}
	return name
}

// metaReadIndexReadTimeout bounds the per-Linearizable-read meta readIndex barrier
// (cluster.Node.metaReadBarrier): the budget a Linearizable read gives the
// coordinator to confirm its LOCAL meta-FSM is caught up to the meta leader's
// verified command frontier before it resolves the catalog (alias + partition gen).
// It mirrors the shard-level data barrier's linearizableReadTimeout (shard/store.go,
// 5s) so the catalog-freshness budget matches the data-freshness budget a
// Linearizable read already accepts. A package-level var so tests can shorten it.
// ZERO cost off the Linearizable path: it is only read inside resolveCollectionForRead
// when rc == ops.ConsistencyLinearizable.
var metaReadIndexReadTimeout = 5 * time.Second

// resolveCollectionForRead is the read-path catalog chokepoint: it linearizes the
// LOCAL catalog read (alias + partition-gen resolution) for a Linearizable read,
// then resolves the alias. For a Linearizable read on a clustered node it runs the
// meta readIndex barrier (cluster.Node.metaReadBarrier) FIRST so the subsequent
// LOCAL catalog reads (resolveAlias here + the caller's e.catalog.PartitionsGen)
// observe every catalog command committed as of a leader-verified read point — this
// closes alias/gen staleness AND create/drop staleness for Linearizable reads. On
// barrier timeout it returns *cluster.ErrMetaLinearizableTimeout and DOES NOT
// resolve/serve (fail loud, never stale).
//
// Cost contract (PERFORMANCE IS A HARD CONSTRAINT):
//   - non-Linearizable rc (AnyReplica/LeaderOnly) → byte-identical to a bare
//     resolveAlias: NO barrier, NO meta RTT, only one rc comparison.
//   - single-node / pure-embedded / RF<=1 → metaReadBarrier short-circuits to a
//     local no-op (n.meta == nil || len(Peers) <= 1), zero forwards.
//   - it is called ONCE per read on the COORDINATOR, BEFORE fan-out — never
//     per-partition — so a P>1 Linearizable read still issues at most one barrier.
//
// WRITES never call this; they keep the plain resolveAlias (no barrier).
// ConsistencyBoundedStaleness deliberately does NOT run the strict meta
// MetaReadBarrier here — it is treated like AnyReplica for the CATALOG read.
// Bounded staleness is a DATA freshness SLO (enforced per-shard by the bound
// guard); the catalog is small and rarely-stale, so forcing a meta barrier would
// add cost without serving the feature's intent. Only ConsistencyLinearizable
// arms the meta barrier below.
func (e *embedded) resolveCollectionForRead(collection string, rc uint8, deadline time.Time) (string, error) {
	if rc == ops.ConsistencyLinearizable && e.node != nil {
		// Run the barrier BEFORE resolving so the local meta-FSM is fresh for BOTH the
		// alias lookup below and the caller's subsequent PartitionsGen read. The barrier
		// itself is a no-op for single-node/no-meta-peers (n.meta == nil ⇒ nil).
		if err := e.node.MetaReadBarrier(deadline); err != nil {
			return "", err
		}
	}
	return e.resolveAlias(collection), nil
}

// cleanupAliasesFor removes every alias whose TARGET == collection (canonical),
// the drop-cascade invoked after a collection is dropped + its catalog entry
// neutralized. It is BEST-EFFORT: a cascade failure must NOT fail the drop, so
// errors are swallowed (a dangling alias resolves to a gone collection →
// downstream not-found, which is safe). The batch of deletes is issued as ONE
// SetAliases call (atomic). The collection name is canonicalized so it matches
// the canonical targets stored in the catalog.
func (e *embedded) cleanupAliasesFor(_ context.Context, collection string) {
	want := ops.CanonicalName(collection)
	var deletes []AliasAction
	for alias, target := range e.catalog.ListAliases() {
		if target == want {
			deletes = append(deletes, AliasAction{Alias: alias, Delete: true})
		}
	}
	if len(deletes) == 0 {
		return
	}
	if err := e.catalog.SetAliases(deletes); err != nil {
		slog.Warn("alias drop-cascade failed (best-effort, ignoring)", "collection", collection, "err", err)
	}
}

// partitionOf returns the physical collection name and true when collection is
// partitioned (P>1); otherwise it returns ("", false) and the caller uses the
// logical name unchanged. The catalog miss for unpartitioned collections is an
// in-process map lookup, so the single-partition hot path adds no Raft traffic.
func (e *embedded) partitionOf(collection string, id uint64) (string, bool) {
	P, gen, ok := e.catalog.PartitionsGen(collection)
	if !ok || P <= 1 {
		return "", false
	}
	return string(ops.PartitionKeyGen(collection, gen, ops.PartitionOf(id, P))), true
}

// dualTargets resolves the physical write target(s) for a point write of id in
// collection. It returns:
//   - live: the physical name on the LIVE (read source-of-truth) generation —
//     identical to partitionOf today; the empty string when the collection is
//     unpartitioned (caller writes the logical name unchanged).
//   - target: the physical name on the NEW (target) generation, set only when a
//     reshard is in progress.
//   - dual: true iff a reshard is in progress (ReshardState Resharding) AND the
//     two targets differ — meaning the write must be applied to BOTH.
//
// Routing rules (dual-write through the WHOLE reshard, before AND after cutover):
//   - Stable (no reshard): dual=false, target="". Callers take the existing
//     single-target path (live, or the logical name when unpartitioned). This is
//     byte-for-byte the pre-reshard behavior — the only added cost is one local
//     catalog ReshardState lookup, a map read like PartitionsGen, no Raft. (When
//     ReshardState reports Stable this returns exactly `e.partitionOf` as before.)
//   - Resharding: live = the CATALOG-current gen's partition (= e.partitionOf =
//     the gen reads currently route to, the PRIMARY write target for ordering).
//     target = the OTHER gen's partition, derived EXPLICITLY from the reshard
//     state's source/target pins, independent of which gen the catalog reports as
//     live:
//   - pre-cutover (catalog gen == OldGen): the other gen is NEW.
//   - post-cutover (catalog gen == NewGen): the other gen is OLD.
//     Because the old and new gens always differ, target != live always holds, so
//     dual stays true for the ENTIRE reshard — the old gen keeps receiving every
//     write even AFTER the cutover flip (the linearizable-catalog invariant). The
//     old collapse-at-cutover (target==live ⇒ dual=false) is GONE.
//   - Resharding but the source pin is missing (OldP<=0, e.g. a reshard begun by
//     pre-upgrade code whose meta entry lacks Source — see metaCatalog.ReshardState
//     backward-compat): fall back to TODAY's behavior — target=new, collapse to
//     single-write once live==new at the cutover. This degrades that one in-flight
//     reshard to the pre-fix behavior rather than misrouting; new reshards carry
//     the pin and get the full dual-write.
//
// Callers (applyDualWrite) apply the op to live FIRST (the read source of truth)
// then target; a target-leg error surfaces so an idempotent client retry
// re-applies both. The same op lands on two physical partition names, so it is
// order-independent and idempotent across both legs.
func (e *embedded) dualTargets(collection string, id uint64) (live string, target string, dual bool) {
	live, _ = e.partitionOf(collection, id)
	st, ok := e.catalog.ReshardState(collection)
	if !ok || st.Status != 1 || st.NewP <= 0 {
		return live, "", false
	}
	newTarget := string(ops.PartitionKeyGen(collection, st.NewGen, ops.PartitionOf(id, st.NewP)))
	if st.OldP <= 0 {
		// Backward-compat: a reshard whose state lacks the old-gen pin (pre-upgrade
		// meta entry). We cannot derive the old gen's partition, so degrade to the
		// pre-fix behavior: write the new gen, collapsing to a single write once the
		// catalog has flipped (live == new). The old gen stops getting writes at the
		// cutover for THIS reshard only — exactly today's behavior, no regression.
		if newTarget == live {
			return live, "", false
		}
		return live, newTarget, true
	}
	oldTarget := string(ops.PartitionKeyGen(collection, st.OldGen, ops.PartitionOf(id, st.OldP)))
	// `live` is whichever gen the catalog currently routes reads to (old pre-cutover,
	// new post-cutover). `target` is the OTHER gen, so the write always lands on
	// BOTH the old and the new physical partitions for the whole reshard.
	if live == oldTarget {
		target = newTarget
	} else {
		// Catalog has flipped to the new gen (or live is otherwise the new gen):
		// the other gen to keep fresh is the OLD gen.
		target = oldTarget
	}
	if target == live {
		// Degenerate: old and new partitions coincide for this id (same P and the
		// reshard only changed gen, hashing the id to the same slot). One distinct
		// partition ⇒ no dual-write needed.
		return live, "", false
	}
	return live, target, true
}

// firstWriteOpts coalesces a variadic WriteOpts tail to a single value: the
// methods that historically took NO opts (delete-by-id, payload mutations, the
// plain VectorInsert/MVAdd/NamedInsert) gain a `...WriteOpts` parameter so EVERY
// existing call site keeps compiling byte-for-byte (no arg → zero value → no
// barrier), while a write-consistency-aware caller passes exactly one. More than
// one is a programmer error; we take the first and ignore the rest.
func firstWriteOpts(opts []WriteOpts) WriteOpts {
	if len(opts) == 0 {
		return WriteOpts{}
	}
	return opts[0]
}

// barrierPhys runs the post-commit write-consistency barrier for a single
// PHYSICAL collection name (a partition name like "default/docs#2", or the bare
// canonical name for an unpartitioned collection) that a write just landed on. It
// is a strict no-op unless opts.wcActive() — guarded by the caller — so the
// default write path (WCF unset, wait=true) does ZERO extra work: it never even
// calls BarrierForShard. When active, it maps the physical name to its shard
// index (the SAME shardOf routing the write used) and barriers that one Raft
// group via cluster.Node.BarrierForShard (itself RF-clamped and a fast no-op at
// RF<=1 / single-node embedded, so WCF is a documented no-op there).
func (e *embedded) barrierPhys(physName string, opts WriteOpts) error {
	return e.node.BarrierForShard(
		e.node.ShardIndexForName(physName),
		opts.WriteConsistencyFactor,
		opts.waitValue(),
		cluster.WriteConsistencyTimeout,
	)
}

// applyDualWrite runs op against the live physical target first (the read source
// of truth) and, when dual is set, against the target generation second. A
// live-leg error short-circuits (nothing was written to either gen yet); a
// target-leg error is returned so an idempotent client retry re-applies both
// legs. op must be idempotent (insert/upsert/delete all are). It returns the
// live leg's body (callers only inspect the live result — e.g. delete's
// existed-flag — which reflects the read source of truth).
//
// Write-consistency: after each physical leg commits at Raft majority, if
// opts.wcActive() the barrier runs for THAT physical target — both gens during a
// reshard dual-write, since the write isn't consistent until BOTH land. When
// opts is the zero value (the default for every existing caller) wcActive() is
// false and NOT A SINGLE extra instruction beyond the wcActive() check runs, so
// the default path is byte/branch-identical to before.
// isCollectionGone reports whether err is a "the physical partition does not
// exist" error from ANY vector family — the signal that a dual-write leg targeted
// a partition that was concurrently DROPPED (e.g. the retiring old generation
// retired by the reshard's Phase-6 cleanup after the cutover). It matches the
// per-family not-found messages by substring (the error is wrapped/prefixed as it
// crosses e.Call and, on the remote path, the client transport — equality would
// not survive that), covering all three families that route through dualTargets:
//
//   - dense (ops):  "ops: unknown collection %q"     (ops/builtin.go, ops/multivector.go)
//   - multi-vector: "vector: no multi-vector collection %q" (vector/collections_multi.go)
//   - named/dense:  "vector: no collection %q"        (vector/collections.go)
//
// It deliberately matches ONLY the partition-gone class. A TRANSIENT failure
// (not-leader, timeout) carries a different message and is NOT matched, so a
// still-alive old gen that merely couldn't be reached surfaces its error and is
// never silently left stale.
func isCollectionGone(err error) bool {
	if err == nil {
		return false
	}
	msg := err.Error()
	return strings.Contains(msg, "unknown collection ") ||
		strings.Contains(msg, "no multi-vector collection ") ||
		strings.Contains(msg, "no collection ")
}

func (e *embedded) applyDualWrite(opName string, live, target string, dual bool, opts WriteOpts, encode func(phys string) []byte) ([]byte, error) {
	body, err := e.Call(context.Background(), opName, encode(live))
	if err != nil {
		return nil, err
	}
	if opts.wcActive() {
		if err := e.barrierPhys(live, opts); err != nil {
			return nil, err
		}
	}
	if dual {
		// targetRetiring: the secondary (target) leg points at the RETIRING OLD
		// generation iff its generation is LOWER than the live leg's. Post-cutover
		// the catalog routes reads to the NEW (higher) gen, so live=new>target=old
		// ⇒ retiring; pre-cutover live=old<target=new ⇒ NOT retiring (the new gen is
		// being built and is authoritative-for-the-future). Derived from the two
		// physical names applyDualWrite already holds, so no flag threads through the
		// ~18 dualTargets call sites.
		targetRetiring := ops.PartitionGenOf(target) < ops.PartitionGenOf(live)
		if _, err := e.Call(context.Background(), opName, encode(target)); err != nil {
			// Best-effort tolerance, scoped TIGHTLY: only when the target is the
			// retiring old gen AND the error is partition-gone. In that case the
			// authoritative live (new-gen) leg already committed and every read routes
			// there, so a write to the just-dropped old-gen partition losing the race
			// with Phase-6 cleanup is harmless — tolerate it. We skip the target-leg
			// write-consistency barrier because nothing landed on that gen.
			//
			// Everything else propagates:
			//   - pre-cutover (target = new gen, NOT retiring): a not-found means data
			//     loss in the gen being built — must surface. (Asserted by
			//     TestEmbeddedDualWriteNewGenLegErrorSurfaces.)
			//   - a TRANSIENT error (not-leader/timeout) on a still-alive old gen: must
			//     surface, else the old gen goes silently stale and a lagging follower
			//     could read non-linearizable data.
			if targetRetiring && isCollectionGone(err) {
				return body, nil
			}
			return nil, err
		}
		if opts.wcActive() {
			if err := e.barrierPhys(target, opts); err != nil {
				return nil, err
			}
		}
	}
	return body, nil
}

func (e *embedded) VectorInsertExt(_ context.Context, collection string, id uint64, vec []float32, opts VectorInsertOpts) error {
	collection = e.resolveAlias(collection)
	live, target, dual := e.dualTargets(collection, id)
	if live != "" {
		collection = live
	}
	_, err := e.applyDualWrite("vector_insert", collection, target, dual, opts.WriteOpts,
		func(phys string) []byte {
			// keyTTL block rides AFTER expectedVersion; empty map = byte-identical to
			// EncodeVectorInsertArgsExt (the prior wire shape for this path).
			return ops.EncodeVectorInsertArgsKeyTTL(phys, id, vec, opts.TTL, opts.Metadata, opts.Sparse, opts.KeyTTLMs)
		})
	return err
}

func (e *embedded) VectorSearchExt(_ context.Context, collection string, query []float32, k int, opts VectorSearchOpts) ([]VectorResult, FanMeta, error) {
	collection, err := e.resolveCollectionForRead(collection, opts.ReadConsistency, time.Now().Add(metaReadIndexReadTimeout))
	if err != nil {
		return nil, FanMeta{}, err
	}
	if P, gen, ok := e.catalog.PartitionsGen(collection); ok && P > 1 {
		res, fr, err := e.denseFanOut(collection, P, gen, query, k, opts.Filter, opts.ReadConsistency, opts.OnPartitionUnavailable, opts.MaxStaleness)
		return res, FanMeta{Degraded: fr.Degraded, Missing: fr.Missing}, err
	}
	body, err := e.callReadLeader("vector_search",
		ops.EncodeVectorSearchArgsOpts(collection, k, query, opts.Filter, opts.ReadConsistency, opts.OnPartitionUnavailable, opts.MaxStaleness),
		opts.ReadConsistency)
	if err != nil {
		return nil, FanMeta{}, err
	}
	res, err := ops.DecodeVectorSearchResults(body)
	return res, FanMeta{}, err
}

func (e *embedded) VectorHybridSearch(_ context.Context, collection string, dense []float32, k int, opts VectorHybridOpts) ([]VectorResult, FanMeta, error) {
	collection, err := e.resolveCollectionForRead(collection, opts.ReadConsistency, time.Now().Add(metaReadIndexReadTimeout))
	if err != nil {
		return nil, FanMeta{}, err
	}
	// Partitioned collections (P>1): fan out vector_hybrid_lanes to every
	// partition, union the per-partition lanes, truncate to the global
	// denseK/sparseK, then fuse ONCE — exact hybrid fan-out.
	if P, gen, ok := e.catalog.PartitionsGen(collection); ok && P > 1 {
		res, fr, err := e.hybridFanOut(collection, P, gen, dense, k, opts)
		return res, FanMeta{Degraded: fr.Degraded, Missing: fr.Missing}, err
	}
	body, err := e.callReadLeader("vector_hybrid_search",
		ops.EncodeHybridSearchArgsOpts(collection, dense, k, opts.Sparse, toVectorHybridOpts(opts), opts.ReadConsistency, opts.OnPartitionUnavailable, opts.MaxStaleness),
		opts.ReadConsistency)
	if err != nil {
		return nil, FanMeta{}, err
	}
	res, err := ops.DecodeHybridResults(body)
	return res, FanMeta{}, err
}

// VectorSearchText runs a BM25 full-text search and returns enriched docs. Under
// partitioning it fans vector_search_text to every partition and merges by
// DESCENDING BM25 score (higher = more relevant) into the global top-k — the
// per-shard-local-IDF approximation (see textDocsFanOut / the design-doc caveat).
func (e *embedded) VectorSearchText(_ context.Context, collection string, query string, k int, opts VectorSearchOpts) ([]VectorDocument, FanMeta, error) {
	collection, err := e.resolveCollectionForRead(collection, opts.ReadConsistency, time.Now().Add(metaReadIndexReadTimeout))
	if err != nil {
		return nil, FanMeta{}, err
	}
	if P, gen, ok := e.catalog.PartitionsGen(collection); ok && P > 1 {
		docs, fr, err := e.textDocsFanOut(collection, P, gen, query, k, opts.Filter, opts.ReadConsistency, opts.OnPartitionUnavailable, opts.MaxStaleness, opts.GlobalIDF)
		return docs, FanMeta{Degraded: fr.Degraded, Missing: fr.Missing}, err
	}
	body, err := e.callReadLeader("vector_search_text",
		ops.EncodeSearchTextArgsOpts(collection, query, k, opts.Filter, opts.ReadConsistency, opts.OnPartitionUnavailable, opts.MaxStaleness),
		opts.ReadConsistency)
	if err != nil {
		return nil, FanMeta{}, err
	}
	docs, err := ops.DecodeVectorDocs(body)
	return docs, FanMeta{}, err
}

// VectorHybridText fuses a dense KNN lane with a BM25 text lane. Under
// partitioning it fans vector_hybrid_text_lanes to every partition, unions the
// per-partition lanes, truncates to the global denseK/sparseK, then fuses ONCE
// (hybridTextFanOut). The text lane's IDF is per-shard-local (approximate).
func (e *embedded) VectorHybridText(_ context.Context, collection string, dense []float32, query string, k int, opts VectorHybridOpts) ([]VectorResult, FanMeta, error) {
	collection, err := e.resolveCollectionForRead(collection, opts.ReadConsistency, time.Now().Add(metaReadIndexReadTimeout))
	if err != nil {
		return nil, FanMeta{}, err
	}
	if P, gen, ok := e.catalog.PartitionsGen(collection); ok && P > 1 {
		res, fr, err := e.hybridTextFanOut(collection, P, gen, dense, query, k, opts)
		return res, FanMeta{Degraded: fr.Degraded, Missing: fr.Missing}, err
	}
	body, err := e.callReadLeader("vector_hybrid_text",
		ops.EncodeHybridTextArgsOpts(collection, dense, query, k, toVectorHybridOpts(opts), opts.ReadConsistency, opts.OnPartitionUnavailable, opts.MaxStaleness),
		opts.ReadConsistency)
	if err != nil {
		return nil, FanMeta{}, err
	}
	res, err := ops.DecodeHybridResults(body)
	return res, FanMeta{}, err
}

// VectorQuery runs the unified Query API (vector_query) — a root + N prefetch
// leaves combined by FUSION or RERANK. specBytes is the marshaled pb.QuerySpec
// carried on the wire to each shard; spec is the decoded engine spec the
// coordinator needs for the fusion/rerank merge (mode, method, per-lane K, root
// orientation). For a partitioned collection (P>1) it fans vector_query to every
// partition and merges per mode (queryFanOut); for a single shard it runs the op
// on the read leader and returns the per-partition handler's final result (for
// P=1 FUSION the handler fused into Fused; for RERANK it reranked into Fused).
func (e *embedded) VectorQuery(_ context.Context, collection string, specBytes []byte, spec vector.QuerySpec, opts ReadOpts) ([]VectorResult, FanMeta, error) {
	collection, err := e.resolveCollectionForRead(collection, opts.ReadConsistency, time.Now().Add(metaReadIndexReadTimeout))
	if err != nil {
		return nil, FanMeta{}, err
	}
	if P, gen, ok := e.catalog.PartitionsGen(collection); ok && P > 1 {
		// RECOMMEND cluster pre-pass: a recommend leaf's example ids may live on OTHER
		// partitions, so resolve them cluster-wide + derive the query vector ONCE on the
		// coordinator + rewrite to a dense spec BEFORE fanning out (the partition
		// handlers reject an un-rewritten recommend leaf). Then fan out the rewritten
		// (partition-invariant) dense spec via the EXISTING queryFanOut and prune the
		// example ids from the merged top-k. A spec without recommend leaves is fanned
		// out verbatim (rewritten=false), byte-identical to before.
		rewSpec, rewBytes, exclude, rewritten, rerr := e.resolveRecommendForFanOut(collection, gen, spec)
		if rerr != nil {
			return nil, FanMeta{}, rerr
		}
		if rewritten {
			spec, specBytes = rewSpec, rewBytes
		}
		// DISCOVER cluster pre-pass: a discover leaf's target/context ids may live on
		// OTHER partitions, so resolve them cluster-wide + EMBED the resolved vectors into
		// the leaf ONCE on the coordinator (clearing the ids) BEFORE fanning out. Each
		// partition then runs the discover execLeaf (DiscoverVecs) over its candidates and
		// the coordinator merges by discover score (score-desc — the orientation-aware
		// merge, since the discover leaf carries ScoreDesc=true). Unlike recommend the leaf
		// stays a LeafDiscover (a custom scorer, NOT a dense rewrite) and there is no
		// example-id exclusion/over-fetch. A spec without id-bearing discover leaves is
		// fanned out verbatim (discRewritten=false). Recommend and discover compose: each
		// pre-pass only touches its own leaf kind.
		discSpec, discBytes, discRewritten, derr := e.resolveDiscoverForFanOut(collection, spec)
		if derr != nil {
			return nil, FanMeta{}, derr
		}
		if discRewritten {
			spec, specBytes = discSpec, discBytes
		}
		res, fr, err := e.queryFanOut(collection, P, gen, specBytes, spec, opts.ReadConsistency, opts.OnPartitionUnavailable, opts.MaxStaleness)
		if err != nil {
			return res, FanMeta{Degraded: fr.Degraded, Missing: fr.Missing}, err
		}
		if rewritten && len(exclude) > 0 {
			// Prune the example ids AFTER the global fuse/rerank merge + re-truncate to the
			// requested k (the over-fetch above made room). Mirrors the single-node
			// excludeExamplesFromResult, applied to the coordinator's flat merged result.
			res = vector.ExcludeExamplesFromResults(res, func(r VectorResult) uint64 { return r.ID }, exclude, queryK(rewSpec)-len(exclude))
		}
		return res, FanMeta{Degraded: fr.Degraded, Missing: fr.Missing}, err
	}
	// RECOMMEND pre-pass for single-shard nested-FUSION: mirrors the P>1 coordinator
	// path (resolveRecommendForFanOut) so P==1 and P>1 produce byte-identical results.
	// Without this, the partition's QueryTreeLanes would resolve the recommend leaf
	// locally and include example ids IN the DBSF/Weighted lane normalization — then
	// prune per-lane before returning. The P>1 path rewrites at the coordinator and
	// prunes the merged result POST-FOLD, so per-id scores differ. By applying the same
	// rewrite here (gen==0, non-partitioned → uses the logical collection name for the
	// config call), the partition receives a dense spec (no recommend leaf), runs
	// QueryTreeLanes without any per-lane prune, and returns the lanes with example ids
	// PRESENT for normalization — then the post-fold prune below removes them, matching
	// the P>1 coordinator flow exactly.
	//
	// Non-nested-FUSION specs (flat FUSION, RERANK) and specs without recommend leaves
	// are unchanged: resolveRecommendForFanOut short-circuits if !SpecHasRecommendLeaves.
	var p1RewSpec vector.QuerySpec
	var p1Exclude map[uint64]struct{}
	var p1Rewritten bool
	if vector.SpecHasNestedFusion(spec) {
		rewSpec, rewBytes, excl, rewritten, rerr := e.resolveRecommendForFanOut(collection, 0, spec)
		if rerr != nil {
			return nil, FanMeta{}, rerr
		}
		if rewritten {
			spec, specBytes = rewSpec, rewBytes
			p1RewSpec, p1Exclude, p1Rewritten = rewSpec, excl, true
		}
	}
	body, err := e.callReadLeader("vector_query",
		ops.EncodeQueryArgs(collection, specBytes, opts.ReadConsistency, opts.OnPartitionUnavailable, opts.MaxStaleness),
		opts.ReadConsistency)
	if err != nil {
		return nil, FanMeta{}, err
	}
	qr, derr := ops.DecodeQueryResult(body)
	if derr != nil {
		return nil, FanMeta{}, derr
	}
	// Single shard: the handler returns a mode-tagged payload. RERANK fills Fused
	// (the reranked top-k); FUSION fills Lanes (the UNFUSED prefetch lanes — the
	// wire never carries the single-node engine's locally-fused Fused). Route the
	// one shard's result through the SAME coordinator merge the P>1 path uses so a
	// single partition fuses/reranks identically to the multi-partition union.
	single := []vector.QueryResult{qr}
	var res []VectorResult
	switch spec.Mode {
	case vector.ModeRerank:
		res = rerankMergeFanOut(single, spec.Root, queryK(spec))
	default: // ModeFusion
		// A nested MULTI-lane FUSION spec ships UNFUSED tree-lanes (handleVectorQuery
		// emits them via SpecHasNestedFusion); the single shard runs the SAME recursive
		// tree fold the P>1 coordinator uses over its one lane list, so P==1 is exactly
		// the single-node engine fold at every FUSION node.
		if vector.SpecHasNestedFusion(spec) {
			var terr error
			res, terr = treeFusionMergeFanOut(single, spec, queryK(spec))
			if terr != nil {
				return nil, FanMeta{}, terr
			}
		} else {
			res = fusionMergeFanOut(single, spec, queryK(spec))
		}
	}
	if p1Rewritten && len(p1Exclude) > 0 {
		// Post-fold recommend exclusion: prune example ids from the fused result and
		// re-truncate to the requested k (the over-fetch in resolveRecommendForFanOut
		// widened spec.K to wantK+len(exclude) so k results survive after pruning).
		res = vector.ExcludeExamplesFromResults(res, func(r VectorResult) uint64 { return r.ID }, p1Exclude, queryK(p1RewSpec)-len(p1Exclude))
	}
	return res, FanMeta{}, nil
}

// VectorQueryGrouped is the GROUPED Query API (spec.GroupBy != ""): the group_by
// generalization of VectorQuery. For a partitioned collection (P>1) it fans the
// grouped query to every partition and groups the merged global ordered pool ONCE
// (queryGroupFanOut, P>1==P1); for a single shard (P<=1) it runs the leaf, decodes the
// UNGROUPED flat result + per-id key map, and runs the SAME merge+group step the
// fan-out uses over the one partition — so P==1 is exactly the single-node grouped
// query (one grouping). The recommend/discover pre-passes do NOT apply (grouping is
// dense leaf-only in v1; a recommend/discover leaf with group_by is rejected at the
// engine via the dense grouped guard).
func (e *embedded) VectorQueryGrouped(_ context.Context, collection string, specBytes []byte, spec vector.QuerySpec, opts ReadOpts) ([]VectorGroup, FanMeta, error) {
	collection, err := e.resolveCollectionForRead(collection, opts.ReadConsistency, time.Now().Add(metaReadIndexReadTimeout))
	if err != nil {
		return nil, FanMeta{}, err
	}
	if P, gen, ok := e.catalog.PartitionsGen(collection); ok && P > 1 {
		groups, fr, gerr := e.queryGroupFanOut(collection, P, gen, specBytes, spec, opts.ReadConsistency, opts.OnPartitionUnavailable, opts.MaxStaleness)
		return groups, FanMeta{Degraded: fr.Degraded, Missing: fr.Missing}, gerr
	}
	body, err := e.callReadLeader("vector_query",
		ops.EncodeQueryArgs(collection, specBytes, opts.ReadConsistency, opts.OnPartitionUnavailable, opts.MaxStaleness),
		opts.ReadConsistency)
	if err != nil {
		return nil, FanMeta{}, err
	}
	qr, keys, derr := ops.DecodeQueryResultGroupedFanOut(body)
	if derr != nil {
		return nil, FanMeta{}, derr
	}
	groupsK := queryK(spec)
	groupSize := spec.GroupSize
	if groupSize <= 0 {
		groupSize = 1
	}
	fetchK := vector.GroupFetchK(groupsK, groupSize, 0)
	groups := groupMergedQueryParts([]queryGroupPart{{qr: qr, keys: keys}}, spec, fetchK, groupsK, groupSize)
	return groups, FanMeta{}, nil
}

func (e *embedded) VectorUpsert(_ context.Context, collection string, id uint64, vec []float32, content string, opts VectorInsertOpts) error {
	collection = e.resolveAlias(collection)
	live, target, dual := e.dualTargets(collection, id)
	if live != "" {
		collection = live
	}
	exp, hasExp := opts.expectedVersion()
	_, err := e.applyDualWrite("vector_upsert", collection, target, dual, opts.WriteOpts,
		func(phys string) []byte {
			// CAS + per-key TTL coexist: the keyTTL block rides AFTER expectedVersion.
			// Empty keyTTLMs + no CAS = byte-identical to EncodeVectorUpsertArgs.
			return ops.EncodeVectorUpsertArgsCASKeyTTL(phys, id, vec, content, opts.TTL, opts.Metadata, opts.Sparse, exp, hasExp, opts.KeyTTLMs)
		})
	return err
}

func (e *embedded) VectorSearchDocs(_ context.Context, collection string, query []float32, k int, opts VectorSearchOpts) ([]VectorDocument, FanMeta, error) {
	collection, err := e.resolveCollectionForRead(collection, opts.ReadConsistency, time.Now().Add(metaReadIndexReadTimeout))
	if err != nil {
		return nil, FanMeta{}, err
	}
	if P, gen, ok := e.catalog.PartitionsGen(collection); ok && P > 1 {
		docs, fr, err := e.docsFanOut(collection, P, gen, query, k, opts.Filter, opts.ReadConsistency, opts.OnPartitionUnavailable, opts.MaxStaleness)
		return docs, FanMeta{Degraded: fr.Degraded, Missing: fr.Missing}, err
	}
	body, err := e.callReadLeader("vector_search_docs",
		ops.EncodeVectorSearchArgsOpts(collection, k, query, opts.Filter, opts.ReadConsistency, opts.OnPartitionUnavailable, opts.MaxStaleness),
		opts.ReadConsistency)
	if err != nil {
		return nil, FanMeta{}, err
	}
	docs, err := ops.DecodeVectorDocs(body)
	return docs, FanMeta{}, err
}

func (e *embedded) VectorDeleteByFilter(_ context.Context, collection string, filter VectorFilter) (int, error) {
	collection = e.resolveAlias(collection)
	// Partitioned collections (P>1): scatter vector_delete_by_filter to every
	// partition (each deletes its disjoint subset), sum the counts, and fail-loud
	// on any unreachable partition (OnUnavailable=Fail; idempotent retry recovers).
	if P, gen, ok := e.catalog.PartitionsGen(collection); ok && P > 1 {
		if filter.IsZero() {
			return 0, vector.ErrEmptyFilter // parity with single-partition; avoid P no-op deletes
		}
		n, _, err := e.deleteByFilterFanOut(collection, P, gen, filter)
		return n, err
	}
	body, err := e.Call(context.Background(), "vector_delete_by_filter", ops.EncodeDeleteByFilterArgs(collection, filter))
	if err != nil {
		return 0, err
	}
	return ops.DecodeDeleteByFilterResult(body)
}

func (e *embedded) VectorSearchGroups(_ context.Context, collection string, query []float32, k int, opts VectorGroupOpts) ([]VectorGroup, FanMeta, error) {
	collection, err := e.resolveCollectionForRead(collection, opts.ReadConsistency, time.Now().Add(metaReadIndexReadTimeout))
	if err != nil {
		return nil, FanMeta{}, err
	}
	// Partitioned collections (P>1): fan out vector_group_candidates to every
	// partition, union the per-partition candidates, truncate to the global
	// top-fetchK by distance, then group ONCE — exact group fan-out.
	if P, gen, ok := e.catalog.PartitionsGen(collection); ok && P > 1 {
		groups, fr, err := e.groupFanOut(collection, P, gen, query, k, opts)
		return groups, FanMeta{Degraded: fr.Degraded, Missing: fr.Missing}, err
	}
	body, err := e.callReadLeader("vector_search_groups",
		ops.EncodeGroupSearchArgsOpts(collection, k, query, opts, opts.ReadConsistency, opts.OnPartitionUnavailable, opts.MaxStaleness),
		opts.ReadConsistency)
	if err != nil {
		return nil, FanMeta{}, err
	}
	groups, err := ops.DecodeGroups(body)
	return groups, FanMeta{}, err
}

func (e *embedded) VectorMVCreateCollection(ctx context.Context, name string, cfg MultiVectorConfig) error {
	if strings.ContainsAny(name, "#@") {
		return fmt.Errorf("vector: collection name %q must not contain reserved characters '#' or '@'", name)
	}
	// Alias-shadow guard: a new MV collection must not take the name of an existing
	// alias (see CreateCollection).
	if _, ok := e.catalog.ResolveAlias(name); ok {
		return fmt.Errorf("vector: collection name %q is already an alias: %w", name, ErrAliasShadowsCollection)
	}
	// Cross-type / re-partition guard (fail-loud, all Partitions values): the
	// catalog is keyed by name and SHARED with the dense family, so a name that
	// is already partitioned — as either type — must never be re-partitioned, or
	// routing would corrupt.
	if _, _, ok := e.catalog.PartitionsGen(name); ok {
		return fmt.Errorf("vector: collection %q is already partitioned", name)
	}
	// Partitioned MV: reject if a dense collection of the same name exists
	// (err==nil means the dense probe found it; a missing dense collection
	// returns a non-nil error).
	if cfg.Partitions > 1 {
		if _, err := e.Call(ctx, "vector_get_config", ops.EncodeGetConfigArgs(name)); err == nil {
			return fmt.Errorf("vector: a dense collection named %q already exists", name)
		}
	}

	// Single-partition path: byte-for-byte unchanged (no catalog write, no extra
	// collections).
	if cfg.Partitions <= 1 {
		_, err := e.Call(ctx, "vector_mv_create_collection", ops.EncodeMVCreateArgs(name, cfg))
		return err
	}

	// Partitioned path: create the logical MV collection (so the logical name
	// still resolves), then one physical single-partition MV collection per
	// partition, and record P in the catalog. Physical collections are
	// themselves single-partition (Partitions reset to 0) so they never recurse.
	P := cfg.Partitions
	// The logical collection is a single-partition existence/resolution marker;
	// the catalog (SetPartitionsGen below) is the sole source of truth for P.
	// Creating it with Partitions=0 keeps a forwarded logical-create from being
	// re-expanded by a remote node's fanout dispatcher (which only expands
	// Partitions>1), which would otherwise race this coordinator's physical-create
	// loop into "already exists".
	physCfg := cfg
	physCfg.Partitions = 0
	if _, err := e.Call(ctx, "vector_mv_create_collection", ops.EncodeMVCreateArgs(name, physCfg)); err != nil {
		return err
	}
	mvDropArgs := func(n string) []byte { return ops.EncodeMVDeleteArgs(n, 0) }
	for p := 0; p < P; p++ {
		phys := string(ops.PartitionKeyGen(name, 0, p))
		if _, err := e.Call(ctx, "vector_mv_create_collection", ops.EncodeMVCreateArgs(phys, physCfg)); err != nil {
			e.rollbackPartialPartitionedCreate(ctx, name, p, "vector_mv_drop_collection", mvDropArgs)
			return fmt.Errorf("create partition %d/%d for %q: %w", p, P, name, err)
		}
	}
	if err := e.catalog.SetPartitionsGen(name, P, 0); err != nil {
		e.rollbackPartialPartitionedCreate(ctx, name, P, "vector_mv_drop_collection", mvDropArgs)
		return err
	}
	return nil
}

// mvDropCollectionFanout drops a partitioned multi-vector collection. See
// fanoutDrop; this passes the MV drop op + MV-delete args encoder.
func (e *embedded) mvDropCollectionFanout(ctx context.Context, name string, P int, gen uint32) error {
	return e.fanoutDrop(ctx, name, P, gen, "vector_mv_drop_collection", func(phys string) []byte {
		return ops.EncodeMVDeleteArgs(phys, 0)
	})
}

// VectorMVDropCollection is an ADMIN op: it drops the REAL collection (name is
// NOT alias-resolved) and then CASCADE-removes every alias that pointed at it
// (best-effort). Dropping an alias name is undefined; the user drops the real
// collection, which clears its aliases.
func (e *embedded) VectorMVDropCollection(ctx context.Context, name string) error {
	if P, gen, ok := e.catalog.PartitionsGen(name); ok && P > 1 {
		if err := e.mvDropCollectionFanout(ctx, name, P, gen); err != nil {
			return err
		}
		e.cleanupAliasesFor(ctx, name)
		return nil
	}
	if _, err := e.Call(ctx, "vector_mv_drop_collection", ops.EncodeMVDeleteArgs(name, 0)); err != nil {
		return err
	}
	e.cleanupAliasesFor(ctx, name)
	return nil
}

func (e *embedded) VectorMVAdd(_ context.Context, name string, docID uint64, tokens [][]float32, meta VectorMetadata, opts ...WriteOpts) error {
	name = e.resolveAlias(name)
	live, target, dual := e.dualTargets(name, docID)
	if live != "" {
		name = live
	}
	wo := firstWriteOpts(opts)
	exp, hasExp := wo.expectedVersion()
	// CAS + per-key TTL coexist: the keyTTL block rides AFTER the base block and
	// BEFORE the CAS block. The OPTIONAL doc-level sparse rides LAST (omitted when
	// nil/zero). Empty keyTTLMs + no CAS + no sparse = byte-identical to EncodeMVAddArgs.
	_, err := e.applyDualWrite("vector_mv_add", name, target, dual, wo,
		func(phys string) []byte {
			return ops.EncodeMVAddArgsCASKeyTTLSparse(phys, docID, tokens, meta, exp, hasExp, wo.KeyTTLMs, wo.Sparse)
		})
	return err
}

func (e *embedded) VectorMVSearch(_ context.Context, name string, query [][]float32, k int, opts MultiSearchOpts) ([]MultiResult, FanMeta, error) {
	name, err := e.resolveCollectionForRead(name, opts.ReadConsistency, time.Now().Add(metaReadIndexReadTimeout))
	if err != nil {
		return nil, FanMeta{}, err
	}
	if P, gen, ok := e.catalog.PartitionsGen(name); ok && P > 1 {
		res, fr, err := e.mvFanOut(name, P, gen, query, k, opts.CandidatesPerToken, opts.Filter, opts.ReadConsistency, opts.OnPartitionUnavailable, opts.MaxStaleness)
		return res, FanMeta{Degraded: fr.Degraded, Missing: fr.Missing}, err
	}
	body, err := e.callReadLeader("vector_mv_search",
		ops.EncodeMVSearchArgsOptsFilter(name, query, k, opts.CandidatesPerToken, opts.ReadConsistency, opts.OnPartitionUnavailable, opts.Filter, opts.MaxStaleness),
		opts.ReadConsistency)
	if err != nil {
		return nil, FanMeta{}, err
	}
	res, err := ops.DecodeMVResults(body)
	return res, FanMeta{}, err
}

// VectorMVScroll lists live multi-vector documents (id + payload, no token
// vectors) matching filter, paging deterministically id-ascending via a
// resume-after-id cursor. Back-compat convenience for VectorMVScrollExt with
// default consistency. Mirrors VectorNamedScroll.
func (e *embedded) VectorMVScroll(ctx context.Context, name string, filter VectorFilter, limit int, cursor string) ([]VectorDocument, FanMeta, string, error) {
	return e.VectorMVScrollExt(ctx, name, filter, limit, cursor, MVScrollOpts{})
}

// VectorMVScrollExt is VectorMVScroll with read-consistency opts. A Linearizable
// scroll runs the meta readIndex barrier on the coordinator, routes a
// single-partition scroll to the leader via callReadLeader (so the per-shard data
// barrier fires), and re-encodes rc into every per-partition arg on the fan-out
// path. Mirrors VectorNamedScrollExt / VectorScroll; the merge is id-ASCENDING
// (scroll has no score). MV doc ids are disjoint across partitions, so one global
// cursor pages the whole collection gap-free + dup-free.
func (e *embedded) VectorMVScrollExt(_ context.Context, name string, filter VectorFilter, limit int, cursor string, opts MVScrollOpts) ([]VectorDocument, FanMeta, string, error) {
	// Decode the input cursor TYPED so the order_by (v2) and id-scroll (v1) paths both
	// validate the cursor version against opts.OrderBy before dispatch (mirror VectorScroll).
	dec, err := ops.DecodeScrollCursorTyped(cursor)
	if err != nil {
		return nil, FanMeta{}, "", err
	}
	var order *ops.ScrollOrder
	var afterID uint64
	var hasAfter bool
	if opts.OrderBy != nil {
		ob := opts.OrderBy
		var verr error
		order, afterID, hasAfter, verr = buildScrollOrder(ob, dec)
		if verr != nil {
			return nil, FanMeta{}, "", verr
		}
	} else {
		// No order_by: only a v1 (id-only) cursor is valid; a v2 cursor here means a
		// client dropped order_by mid-pagination — reject loud (symmetric guard).
		if dec.Present && dec.Version != 1 {
			return nil, FanMeta{}, "", ops.ErrCursorOrderMismatch
		}
		afterID, hasAfter = dec.LastID, dec.Present
	}
	name, err = e.resolveCollectionForRead(name, opts.ReadConsistency, time.Now().Add(metaReadIndexReadTimeout))
	if err != nil {
		return nil, FanMeta{}, "", err
	}
	if P, gen, ok := e.catalog.PartitionsGen(name); ok && P > 1 {
		docs, fr, nextCursor, ferr := e.mvScrollFanOut(name, P, gen, filter, limit, opts.ReadConsistency, opts.OnPartitionUnavailable, afterID, hasAfter, order, opts.MaxStaleness)
		return docs, FanMeta{Degraded: fr.Degraded, Missing: fr.Missing}, nextCursor, ferr
	}
	// Unpartitioned single-shard path: the args carry rc; callReadLeader routes a
	// LeaderOnly/Linearizable scroll to the leader so the readIndex barrier runs.
	body, err := e.callReadLeader("vector_mv_scroll",
		ops.EncodeMVScrollArgsOrder(name, filter, limit, opts.ReadConsistency, opts.OnPartitionUnavailable, afterID, hasAfter, order),
		opts.ReadConsistency)
	if err != nil {
		return nil, FanMeta{}, "", err
	}
	docs, err := ops.DecodeVectorDocs(body)
	if err != nil {
		return nil, FanMeta{}, "", err
	}
	return docs, FanMeta{}, scrollNextCursorOrder(docs, limit, opts.OrderBy), nil
}

func (e *embedded) VectorMVDelete(_ context.Context, name string, docID uint64, opts ...WriteOpts) (bool, error) {
	name = e.resolveAlias(name)
	live, target, dual := e.dualTargets(name, docID)
	if live != "" {
		name = live
	}
	// Dual-delete removes from both gens; the returned existed-flag reflects the
	// live (read source-of-truth) gen.
	wo := firstWriteOpts(opts)
	exp, hasExp := wo.expectedVersion()
	body, err := e.applyDualWrite("vector_mv_delete", name, target, dual, wo,
		func(phys string) []byte { return ops.EncodeMVDeleteArgsCAS(phys, docID, exp, hasExp) })
	if err != nil {
		return false, err
	}
	return len(body) > 0 && body[0] == 1, nil
}

// VectorMVAddIfAbsent routes to the live gen's physical partition (same as
// VectorMVAdd) and runs the atomic MV add-if-absent op there.
func (e *embedded) VectorMVAddIfAbsent(_ context.Context, name string, docID uint64, tokens [][]float32, meta VectorMetadata) (bool, error) {
	name = e.resolveAlias(name)
	if phys, ok := e.partitionOf(name, docID); ok {
		name = phys
	}
	body, err := e.Call(context.Background(), "vector_mv_add_if_absent", ops.EncodeMVAddArgs(name, docID, tokens, meta))
	if err != nil {
		return false, err
	}
	return ops.DecodeIfAbsentResult(body)
}

// VectorMVExists probes the live gen's physical partition for docID's liveness.
func (e *embedded) VectorMVExists(_ context.Context, name string, docID uint64) (bool, error) {
	name = e.resolveAlias(name)
	if phys, ok := e.partitionOf(name, docID); ok {
		name = phys
	}
	body, err := e.Call(context.Background(), "vector_mv_exists", ops.EncodeMVExistsArgs(name, docID))
	if err != nil {
		return false, err
	}
	return ops.DecodeExistsResult(body)
}

// VectorMVGet retrieves a multi-vector document by id from the live gen's owning
// physical partition (route-by-id). not-found is the decoded found=0 FLAG.
// VectorMVGet is the back-compat convenience form (AnyReplica); see VectorGet.
func (e *embedded) VectorMVGet(ctx context.Context, name string, docID uint64, withVector, withPayload bool) (bool, [][]float32, VectorMetadata, error) {
	return e.VectorMVGetExt(ctx, name, docID, withVector, withPayload, ReadOpts{})
}

// VectorMVGetExt retrieves a multi-vector document by id with read-consistency:
// route-by-id to the owning partition's leader + readIndex barrier for a
// Linearizable read. rc==0 is the legacy plain-Call path. See VectorGetExt.
func (e *embedded) VectorMVGetExt(_ context.Context, name string, docID uint64, withVector, withPayload bool, opts ReadOpts) (bool, [][]float32, VectorMetadata, error) {
	name, err := e.resolveCollectionForRead(name, opts.ReadConsistency, time.Now().Add(metaReadIndexReadTimeout))
	if err != nil {
		return false, nil, nil, err
	}
	if phys, ok := e.partitionOf(name, docID); ok {
		name = phys
	}
	body, err := e.callReadLeader("vector_mv_get",
		ops.EncodeVectorGetArgsOpts(name, docID, getFlags(withVector, withPayload), opts.ReadConsistency, opts.OnPartitionUnavailable, opts.MaxStaleness),
		opts.ReadConsistency)
	if err != nil {
		return false, nil, nil, err
	}
	return ops.DecodeMVGetResult(body)
}

// VectorMVGetBatch retrieves many multi-vector documents by id in one op. For an
// unpartitioned (or P<=1) collection it is a SINGLE vector_mv_get_batch Call with
// all (deduped) ids. For a partitioned collection it groups the ids by their
// owning partition and asks each partition ONLY for its owned subset concurrently
// (scatterIDsByPartition), then merges. A partial miss is normal (the absent ids
// land in `missing`, never an error); points + missing are returned sorted
// ascending by id. Like VectorMVGet this is AnyReplica (no read consistency) and
// route-by-id correct. MV has NO ttl. The MV clone of VectorNamedGetBatch.
func (e *embedded) VectorMVGetBatch(_ context.Context, collection string, ids []uint64, withVector, withPayload bool) ([]MVBatchGetPoint, []uint64, error) {
	collection = e.resolveAlias(collection)
	flags := getFlags(withVector, withPayload)
	ids = dedupIDs(ids)
	if len(ids) == 0 {
		return nil, nil, nil
	}

	P, gen, ok := e.catalog.PartitionsGen(collection)
	if !ok || P <= 1 {
		// Unpartitioned / single-partition fast path: ONE call with all ids.
		body, err := e.Call(context.Background(), "vector_mv_get_batch", ops.EncodeVectorGetBatchArgs(collection, ids, flags))
		if err != nil {
			return nil, nil, err
		}
		rows, err := ops.DecodeMVGetBatchResult(body)
		if err != nil {
			return nil, nil, err
		}
		return splitMVBatchRows(rows)
	}
	all, err := scatterIDsByPartition(e, collection, P, gen, ids, func(phys string, sub []uint64) ([]ops.MVGetBatchRow, error) {
		raw, err := e.node.CallPhysical(phys, "vector_mv_get_batch", ops.EncodeVectorGetBatchArgs(phys, sub, flags), false)
		if err != nil {
			return nil, err
		}
		return ops.DecodeMVGetBatchResult(raw)
	})
	if err != nil {
		return nil, nil, err
	}
	return splitMVBatchRows(all)
}

// VectorMVSetPayload merges patch into the doc's payload on the owning partition,
// dual-writing during a reshard (mirror VectorMVDelete / VectorSetPayload).
func (e *embedded) VectorMVSetPayload(_ context.Context, name string, docID uint64, patch VectorMetadata, keyTTLMs map[string]int64, opts ...WriteOpts) (bool, error) {
	name = e.resolveAlias(name)
	live, target, dual := e.dualTargets(name, docID)
	if live != "" {
		name = live
	}
	wo := firstWriteOpts(opts)
	exp, hasExp := wo.expectedVersion()
	body, err := e.applyDualWrite("vector_mv_set_payload", name, target, dual, wo,
		func(phys string) []byte {
			return ops.EncodeSetPayloadArgsCAS(phys, docID, patch, keyTTLMs, exp, hasExp)
		})
	if err != nil {
		return false, err
	}
	return ops.DecodePayloadResult(body)
}

func (e *embedded) VectorMVOverwritePayload(_ context.Context, name string, docID uint64, meta VectorMetadata, keyTTLMs map[string]int64, opts ...WriteOpts) (bool, error) {
	name = e.resolveAlias(name)
	live, target, dual := e.dualTargets(name, docID)
	if live != "" {
		name = live
	}
	wo := firstWriteOpts(opts)
	exp, hasExp := wo.expectedVersion()
	body, err := e.applyDualWrite("vector_mv_overwrite_payload", name, target, dual, wo,
		func(phys string) []byte { return ops.EncodeSetPayloadArgsCAS(phys, docID, meta, keyTTLMs, exp, hasExp) })
	if err != nil {
		return false, err
	}
	return ops.DecodePayloadResult(body)
}

func (e *embedded) VectorMVDeletePayloadKeys(_ context.Context, name string, docID uint64, keys []string, opts ...WriteOpts) (bool, error) {
	name = e.resolveAlias(name)
	live, target, dual := e.dualTargets(name, docID)
	if live != "" {
		name = live
	}
	wo := firstWriteOpts(opts)
	exp, hasExp := wo.expectedVersion()
	body, err := e.applyDualWrite("vector_mv_delete_payload_keys", name, target, dual, wo,
		func(phys string) []byte { return ops.EncodeDeletePayloadKeysArgsCAS(phys, docID, keys, exp, hasExp) })
	if err != nil {
		return false, err
	}
	return ops.DecodePayloadResult(body)
}

func (e *embedded) VectorMVClearPayload(_ context.Context, name string, docID uint64, opts ...WriteOpts) (bool, error) {
	name = e.resolveAlias(name)
	live, target, dual := e.dualTargets(name, docID)
	if live != "" {
		name = live
	}
	wo := firstWriteOpts(opts)
	exp, hasExp := wo.expectedVersion()
	body, err := e.applyDualWrite("vector_mv_clear_payload", name, target, dual, wo,
		func(phys string) []byte { return ops.EncodeClearPayloadArgsCAS(phys, docID, exp, hasExp) })
	if err != nil {
		return false, err
	}
	return ops.DecodePayloadResult(body)
}

// VectorNamedGet retrieves a named-vector point by id from the live gen's owning
// physical partition (route-by-id). not-found is the decoded found=0 FLAG.
// VectorNamedGet is the back-compat convenience form (AnyReplica); see VectorGet.
func (e *embedded) VectorNamedGet(ctx context.Context, name string, id uint64, withVector, withPayload bool) (bool, map[string][]float32, VectorMetadata, time.Duration, error) {
	return e.VectorNamedGetExt(ctx, name, id, withVector, withPayload, ReadOpts{})
}

// VectorNamedGetExt retrieves a named-vector point by id with read-consistency:
// route-by-id to the owning partition's leader + readIndex barrier for a
// Linearizable read. rc==0 is the legacy plain-Call path. See VectorGetExt.
func (e *embedded) VectorNamedGetExt(_ context.Context, name string, id uint64, withVector, withPayload bool, opts ReadOpts) (bool, map[string][]float32, VectorMetadata, time.Duration, error) {
	name, err := e.resolveCollectionForRead(name, opts.ReadConsistency, time.Now().Add(metaReadIndexReadTimeout))
	if err != nil {
		return false, nil, nil, 0, err
	}
	if phys, ok := e.partitionOf(name, id); ok {
		name = phys
	}
	body, err := e.callReadLeader("vector_named_get",
		ops.EncodeVectorGetArgsOpts(name, id, getFlags(withVector, withPayload), opts.ReadConsistency, opts.OnPartitionUnavailable, opts.MaxStaleness),
		opts.ReadConsistency)
	if err != nil {
		return false, nil, nil, 0, err
	}
	return ops.DecodeNamedGetResult(body)
}

// VectorNamedGetBatch retrieves many named-vector points by id in one op. For an
// unpartitioned (or P<=1) collection it is a SINGLE vector_named_get_batch Call
// with all (deduped) ids. For a partitioned collection it groups the ids by their
// owning partition and asks each partition ONLY for its owned subset concurrently
// (scatterIDsByPartition), then merges. A partial miss is normal (the absent ids
// land in `missing`, never an error); points + missing are returned sorted
// ascending by id. Like VectorNamedGet this is AnyReplica (no read consistency)
// and route-by-id correct. The named clone of VectorGetBatch.
func (e *embedded) VectorNamedGetBatch(_ context.Context, collection string, ids []uint64, withVector, withPayload bool) ([]NamedBatchGetPoint, []uint64, error) {
	collection = e.resolveAlias(collection)
	flags := getFlags(withVector, withPayload)
	ids = dedupIDs(ids)
	if len(ids) == 0 {
		return nil, nil, nil
	}

	P, gen, ok := e.catalog.PartitionsGen(collection)
	if !ok || P <= 1 {
		// Unpartitioned / single-partition fast path: ONE call with all ids.
		body, err := e.Call(context.Background(), "vector_named_get_batch", ops.EncodeVectorGetBatchArgs(collection, ids, flags))
		if err != nil {
			return nil, nil, err
		}
		rows, err := ops.DecodeNamedGetBatchResult(body)
		if err != nil {
			return nil, nil, err
		}
		return splitNamedBatchRows(rows)
	}
	all, err := scatterIDsByPartition(e, collection, P, gen, ids, func(phys string, sub []uint64) ([]ops.NamedGetBatchRow, error) {
		raw, err := e.node.CallPhysical(phys, "vector_named_get_batch", ops.EncodeVectorGetBatchArgs(phys, sub, flags), false)
		if err != nil {
			return nil, err
		}
		return ops.DecodeNamedGetBatchResult(raw)
	})
	if err != nil {
		return nil, nil, err
	}
	return splitNamedBatchRows(all)
}

// VectorNamedSetPayload merges patch into the point's shared payload on the owning
// partition, dual-writing during a reshard (mirror VectorNamedDelete / VectorSetPayload).
func (e *embedded) VectorNamedSetPayload(_ context.Context, name string, id uint64, patch VectorMetadata, keyTTLMs map[string]int64, opts ...WriteOpts) (bool, error) {
	name = e.resolveAlias(name)
	live, target, dual := e.dualTargets(name, id)
	if live != "" {
		name = live
	}
	wo := firstWriteOpts(opts)
	exp, hasExp := wo.expectedVersion()
	body, err := e.applyDualWrite("vector_named_set_payload", name, target, dual, wo,
		func(phys string) []byte { return ops.EncodeSetPayloadArgsCAS(phys, id, patch, keyTTLMs, exp, hasExp) })
	if err != nil {
		return false, err
	}
	return ops.DecodePayloadResult(body)
}

func (e *embedded) VectorNamedOverwritePayload(_ context.Context, name string, id uint64, meta VectorMetadata, keyTTLMs map[string]int64, opts ...WriteOpts) (bool, error) {
	name = e.resolveAlias(name)
	live, target, dual := e.dualTargets(name, id)
	if live != "" {
		name = live
	}
	wo := firstWriteOpts(opts)
	exp, hasExp := wo.expectedVersion()
	body, err := e.applyDualWrite("vector_named_overwrite_payload", name, target, dual, wo,
		func(phys string) []byte { return ops.EncodeSetPayloadArgsCAS(phys, id, meta, keyTTLMs, exp, hasExp) })
	if err != nil {
		return false, err
	}
	return ops.DecodePayloadResult(body)
}

func (e *embedded) VectorNamedDeletePayloadKeys(_ context.Context, name string, id uint64, keys []string, opts ...WriteOpts) (bool, error) {
	name = e.resolveAlias(name)
	live, target, dual := e.dualTargets(name, id)
	if live != "" {
		name = live
	}
	wo := firstWriteOpts(opts)
	exp, hasExp := wo.expectedVersion()
	body, err := e.applyDualWrite("vector_named_delete_payload_keys", name, target, dual, wo,
		func(phys string) []byte { return ops.EncodeDeletePayloadKeysArgsCAS(phys, id, keys, exp, hasExp) })
	if err != nil {
		return false, err
	}
	return ops.DecodePayloadResult(body)
}

func (e *embedded) VectorNamedClearPayload(_ context.Context, name string, id uint64, opts ...WriteOpts) (bool, error) {
	name = e.resolveAlias(name)
	live, target, dual := e.dualTargets(name, id)
	if live != "" {
		name = live
	}
	wo := firstWriteOpts(opts)
	exp, hasExp := wo.expectedVersion()
	body, err := e.applyDualWrite("vector_named_clear_payload", name, target, dual, wo,
		func(phys string) []byte { return ops.EncodeClearPayloadArgsCAS(phys, id, exp, hasExp) })
	if err != nil {
		return false, err
	}
	return ops.DecodePayloadResult(body)
}

// Named-vector (Qdrant-style multi-vector-space) collection fan-out lives in
// named_fanout.go: create re-expands physical partitions, insert routes by id,
// search/search_docs/scroll scatter+union, delete fans out, drop drops every
// partition. Unpartitioned named ops pass straight through to the single
// in-process handler.

func (e *embedded) VectorScroll(_ context.Context, collection string, filter VectorFilter, limit int, opts VectorScrollOpts) ([]VectorDocument, FanMeta, string, error) {
	// Decode the input cursor up front (fail loud on a malformed token before any
	// dispatch). For the id-scroll path afterID/hasAfter is the global resume lower
	// bound; for order_by it is the (value, id) lower bound. The cursor version must
	// agree with whether order_by is set — reject loud on a mismatch.
	dec, err := ops.DecodeScrollCursorTyped(opts.Cursor)
	if err != nil {
		return nil, FanMeta{}, "", err
	}
	var order *ops.ScrollOrder
	var afterID uint64
	var hasAfter bool
	if opts.OrderBy != nil {
		ob := opts.OrderBy
		// buildScrollOrder validates the cursor's version/direction/key against the
		// request (v2 for numeric/datetime, v3 for string; a mismatch ⇒ reject loud)
		// and maps the resume (value/string, id) onto the order block + args afterID.
		// The coordinator itself does not seek (the leaf/engine does), so no afterKey
		// local is needed here.
		var verr error
		order, afterID, hasAfter, verr = buildScrollOrder(ob, dec)
		if verr != nil {
			return nil, FanMeta{}, "", verr
		}
	} else {
		// No order_by: only a v1 (id-only) cursor is valid. A v2 cursor here means a
		// client dropped order_by mid-pagination — the id-scroll path cannot honor an
		// ordered resume, so reject loud (symmetric to ValidateOrderCursor).
		if dec.Present && dec.Version != 1 {
			return nil, FanMeta{}, "", ops.ErrCursorOrderMismatch
		}
		afterID = dec.LastID
		hasAfter = dec.Present
	}
	collection, err = e.resolveCollectionForRead(collection, opts.ReadConsistency, time.Now().Add(metaReadIndexReadTimeout))
	if err != nil {
		return nil, FanMeta{}, "", err
	}
	if P, gen, ok := e.catalog.PartitionsGen(collection); ok && P > 1 {
		docs, fr, nextCursor, ferr := e.scrollFanOut(collection, P, gen, filter, limit, opts.ReadConsistency, opts.OnPartitionUnavailable, afterID, hasAfter, order, opts.MaxStaleness)
		return docs, FanMeta{Degraded: fr.Degraded, Missing: fr.Missing}, nextCursor, ferr
	}
	// Unpartitioned single-shard path: pass the cursor (incl. order block) straight
	// to the handler. callReadLeader routes a LeaderOnly/Linearizable scroll to the
	// leader so the readIndex barrier runs.
	body, err := e.callReadLeader("vector_scroll",
		ops.EncodeScrollArgsOrder(collection, filter, limit, opts.ReadConsistency, opts.OnPartitionUnavailable, afterID, hasAfter, order),
		opts.ReadConsistency)
	if err != nil {
		return nil, FanMeta{}, "", err
	}
	docs, err := ops.DecodeVectorDocs(body)
	if err != nil {
		return nil, FanMeta{}, "", err
	}
	// next_cursor: a FULL page (len==limit>0) may have more — encode the resume
	// position of the last doc; a short/unlimited page is exhausted ⇒ "".
	nextCursor := scrollNextCursorOrder(docs, limit, opts.OrderBy)
	return docs, FanMeta{}, nextCursor, nil
}

// scrollNextCursor derives the next-page cursor from a returned (id-ascending)
// scroll page: the encoded largest id IFF the page is full (len==limit>0, so more
// may exist), else "" (exhausted — a short or unlimited page has no next page).
// The fan-out coordinator and the unpartitioned path share this rule so the
// exhaustion semantics are identical across backends. This is the v1 (id-only) form;
// scrollNextCursorOrder dispatches to it when order_by is absent.
func scrollNextCursor(docs []VectorDocument, limit int) string {
	if limit > 0 && len(docs) == limit {
		return ops.EncodeScrollCursor(docs[len(docs)-1].ID)
	}
	return ""
}

// scrollNextCursorOrder derives the next-page cursor for an order_by scroll: a v2
// (value, id) token of the last doc IFF the page is full (more may exist), else ""
// (exhausted). When order is nil it falls back to scrollNextCursor (v1 id-only) so
// the no-order_by path is byte-identical. The last doc's order VALUE is read from its
// Metadata via vector.OrderKey (the engine guarantees every returned order_by doc has
// the order field — missing-field points are EXCLUDED at the engine, so OrderKey
// always succeeds here; if it ever fails the page is treated as exhausted rather than
// emitting a corrupt cursor).
func scrollNextCursorOrder(docs []VectorDocument, limit int, order *vector.OrderBy) string {
	if order == nil {
		return scrollNextCursor(docs, limit)
	}
	if limit <= 0 || len(docs) != limit {
		return ""
	}
	last := docs[len(docs)-1]
	if len(order.Tail) > 0 {
		// MULTI-KEY: emit a v4 (k1,…,kN, id) tuple cursor of the last doc. Read each key's
		// value from the last doc's Metadata by its kind; every returned multi-key doc
		// carries every order field (missing-field points are EXCLUDED at the engine). If a
		// key is somehow absent or an encode fails, treat the page as exhausted (return "")
		// rather than emit a corrupt cursor.
		keys := vector.OrderKeyList(order)
		tuple := make([]ops.OrderKeyVal, len(keys))
		for i := range keys {
			if keys[i].Kind == vector.OrderString {
				sk, ok := vector.OrderStringKey(last.Metadata, keys[i].Key)
				if !ok {
					return ""
				}
				tuple[i] = ops.OrderKeyVal{Str: sk, Kind: byte(vector.OrderString)}
				continue
			}
			k, ok := vector.OrderKey(last.Metadata, keys[i].Key, keys[i].IsDatetime)
			if !ok {
				return ""
			}
			tuple[i] = ops.OrderKeyVal{Num: k, Kind: byte(keys[i].Kind)}
		}
		tok, err := ops.EncodeScrollCursorOrderTuple(tuple, last.ID, order.Desc, vector.OrderKeyListHash(keys))
		if err != nil {
			return ""
		}
		return tok
	}
	if order.Kind == vector.OrderString {
		sk, ok := vector.OrderStringKey(last.Metadata, order.Key)
		if !ok {
			return "" // defensive: every returned string order_by doc carries the field
		}
		tok, err := ops.EncodeScrollCursorOrderString(sk, last.ID, order.Desc, vector.OrderKeyHash(order.Key))
		if err != nil {
			return "" // defensive: an oversized resume value cannot be a valid cursor
		}
		return tok
	}
	key, ok := vector.OrderKey(last.Metadata, order.Key, order.IsDatetime)
	if !ok {
		return "" // defensive: every returned order_by doc carries the order field
	}
	return ops.EncodeScrollCursorOrder(key, last.ID, order.Desc, vector.OrderKeyHash(order.Key))
}

// VectorResplit performs the offline generational repartition. Cutover ordering is
// the correctness crux: the new generation's partitions are FULLY BUILT before the
// catalog flip (so all reads/point-ops route to the intact old generation until
// then), the flip is ONE atomic catalog write, and the old generation is dropped
// LAST (orphan-safe — nothing routes there post-flip). If resplit fails before the
// flip, the collection is fully intact (the new-gen partitions are orphans, a
// documented retry/cleanup concern). OFFLINE: the caller MUST quiesce writes first.
func (e *embedded) VectorResplit(ctx context.Context, collection string, newP int) error {
	if newP <= 1 || newP > maxResplitPartitions {
		return fmt.Errorf("resplit: newP must be in [2, %d], got %d", maxResplitPartitions, newP)
	}
	oldP, oldGen, ok := e.catalog.PartitionsGen(collection)
	if !ok || oldP <= 1 {
		return fmt.Errorf("resplit: %q is not partitioned", collection)
	}
	if newP == oldP {
		return nil // no-op
	}
	newGen := oldGen + 1
	// 1. Read existing config from an old-gen physical partition.
	cfgBody, err := e.Call(ctx, "vector_get_config", ops.EncodeGetConfigArgs(string(ops.PartitionKeyGen(collection, oldGen, 0))))
	if err != nil {
		return fmt.Errorf("resplit: get config: %w", err)
	}
	cfg, err := ops.DecodeGetConfigResult(cfgBody)
	if err != nil {
		return fmt.Errorf("resplit: decode config: %w", err)
	}
	cfg.Partitions = 0 // physical partitions are single-partition
	// Self-heal: a prior resplit attempt that failed before the catalog flip may
	// have left newGen partitions behind. Drop them first (no-op if absent) so the
	// create loop below starts clean and a retry succeeds.
	for p := 0; p < newP; p++ {
		phys := string(ops.PartitionKeyGen(collection, newGen, p))
		if _, err := e.Call(ctx, "vector_drop_collection", ops.EncodeDropCollectionArgs(phys)); err != nil {
			return fmt.Errorf("resplit: pre-create cleanup of partition %d: %w", p, err)
		}
	}
	// 2. Create the new generation's physical partitions.
	for p := 0; p < newP; p++ {
		phys := string(ops.PartitionKeyGen(collection, newGen, p))
		if _, err := e.Call(ctx, "vector_create_collection", ops.EncodeCreateCollectionArgs(phys, cfg)); err != nil {
			return fmt.Errorf("resplit: create new partition %d: %w", p, err)
		}
	}
	// 3. Stream every vector old gen -> new gen, re-hashed by newP.
	//
	// COMMIT BATCHING: the naive copy did one Raft-committed vector_insert PER
	// vector, so resplit throughput was commit-bound (~one commit/vector). Instead
	// we split each record into two classes:
	//   - PRISTINE (no metadata/sparse, version <= 1, no TTL, no per-key deadlines):
	//     re-inserting these is a plain fresh insert (version → 1), exactly what the
	//     bulk-load path (vector_bulk_stage += vector_bulk_build) produces. We STAGE
	//     them into their target new partition across all old-partition scans (StageBulk
	//     appends), then BUILD each new partition once — turning N per-vector commits
	//     into O(oldP*newP) stage commits + newP build commits.
	//   - RICH (carries metadata/sparse/version>1/TTL/per-key deadlines): these need the
	//     version- and deadline-PRESERVING vector_insert, so they take the per-record
	//     path VERBATIM — but only AFTER the bulk build (BuildStaged requires an empty
	//     index), and there are typically far fewer of them.
	// Both paths are exact; only the commit count changes.
	type richRec struct {
		phys string
		rec  vector.ScanRecord
	}
	var rich []richRec
	for p := 0; p < oldP; p++ {
		oldPhys := string(ops.PartitionKeyGen(collection, oldGen, p))
		body, err := e.Call(ctx, "vector_scan_vectors", ops.EncodeScanVectorsArgs(oldPhys))
		if err != nil {
			return fmt.Errorf("resplit: scan old partition %d: %w", p, err)
		}
		recs, err := ops.DecodeScanVectorsResult(body)
		if err != nil {
			return fmt.Errorf("resplit: decode scan %d: %w", p, err)
		}
		// Group this scan's pristine records by target partition for one bulk_stage
		// call per (scan, partition); rich records are deferred to the per-record pass.
		stageIDs := make([][]uint64, newP)
		stageVecs := make([][][]float32, newP)
		for _, r := range recs {
			np := ops.PartitionOf(r.ID, newP)
			newPhys := string(ops.PartitionKeyGen(collection, newGen, np))
			if len(r.Metadata) == 0 && r.Sparse == nil && r.Version <= 1 && r.TTL == 0 && len(r.KeyExpires) == 0 {
				stageIDs[np] = append(stageIDs[np], r.ID)
				stageVecs[np] = append(stageVecs[np], r.Vec)
				continue
			}
			rich = append(rich, richRec{phys: newPhys, rec: r})
		}
		for np := 0; np < newP; np++ {
			if len(stageIDs[np]) == 0 {
				continue
			}
			newPhys := string(ops.PartitionKeyGen(collection, newGen, np))
			args, err := ops.EncodeBulkStageArgs(newPhys, stageIDs[np], stageVecs[np])
			if err != nil {
				// These vectors come from the collection's own pristine records, so
				// they are uniform by construction; a failure here means the source
				// index holds a wrong-length vector and resplitting it would corrupt
				// the new partition.
				return fmt.Errorf("resplit: encode bulk-stage for partition %d: %w", np, err)
			}
			if _, err := e.Call(ctx, "vector_bulk_stage", args); err != nil {
				return fmt.Errorf("resplit: bulk-stage partition %d: %w", np, err)
			}
		}
	}
	// Build each new partition's staged pristine vectors in one concurrent pass
	// (the index is still empty — no rich record has been inserted yet).
	for np := 0; np < newP; np++ {
		newPhys := string(ops.PartitionKeyGen(collection, newGen, np))
		if _, err := e.Call(ctx, "vector_bulk_build", ops.EncodeBulkBuildArgs(newPhys, 0)); err != nil {
			return fmt.Errorf("resplit: bulk-build partition %d: %w", np, err)
		}
	}
	// Rich records: version/deadline-preserving per-record reinsert (post-build).
	// ScanRecord.Sparse is *SparseVector; the encoder takes a value, so deref.
	for _, x := range rich {
		var sparse vector.SparseVector
		if x.rec.Sparse != nil {
			sparse = *x.rec.Sparse
		}
		if _, err := e.Call(ctx, "vector_insert", ops.EncodeVectorInsertArgsVersionedKeyExpires(x.phys, x.rec.ID, x.rec.Vec, x.rec.TTL, x.rec.Metadata, sparse, x.rec.Version, x.rec.KeyExpires)); err != nil {
			return fmt.Errorf("resplit: re-insert id %d: %w", x.rec.ID, err)
		}
	}
	// 4. Atomic cutover: flip catalog to {newP, newGen}. After this, all routing
	// uses the new generation.
	if err := e.catalog.SetPartitionsGen(collection, newP, newGen); err != nil {
		return fmt.Errorf("resplit: catalog flip: %w", err)
	}
	// 5. Drop the old generation's partitions (orphan-safe: nothing routes there
	// post-flip).
	for p := 0; p < oldP; p++ {
		oldPhys := string(ops.PartitionKeyGen(collection, oldGen, p))
		if _, err := e.Call(ctx, "vector_drop_collection", ops.EncodeDropCollectionArgs(oldPhys)); err != nil {
			return fmt.Errorf("resplit: cutover done but dropping old partition %d failed: %w", p, err)
		}
	}
	return nil
}

// reshardDrainGrace is the pause between turning on dual-write and starting the
// copy, and again — AFTER the all-nodes-applied cutover gate
// has passed — a SMALL settle window before clearing the reshard status.
//
// Its Phase-2 role: it must exceed in-flight single-leg write latency so any write
// that began BEFORE dual-write was switched on (and therefore landed on the old
// gen only) has fully committed before the copy scans the old gen — the copy then
// carries it to the new gen.
//
// Its Phase-5 role is now SUBSUMED by the Phase-4.5 gate (waitAllNodesCatalogGen),
// which provides the real all-nodes-routed-to-new-gen guarantee instead of a fixed
// blind sleep. The grace is RETAINED (not removed) only as a tiny settle window for
// a request that was already in-flight ON a node AT THE INSTANT that node applied
// the cutover — its catalog read could have picked the OLD gen microseconds before
// the flip while its dual-write leg is still committing — so we let those last
// single-leg-old writes settle before dual-write stops. It is a package-level var
// (not a const) so tests can lower it to keep the concurrent stress test fast while
// non-zero.
var reshardDrainGrace = 2 * time.Second

// reshardCutoverGateTimeout bounds the Phase-4.5 all-nodes-applied cutover gate
// (waitAllNodesCatalogGen). It is generous: the gate is the real correctness
// mechanism that lets a lagging meta-follower keep reading the still-fresh,
// still-dual-written old gen until EVERY node routes to the new gen. A node simply
// replicating one catalog log entry converges in milliseconds; this budget tolerates
// a slow/briefly-stalled follower. On timeout the reshard does NOT fail — it LOGS
// the unconfirmed nodes and PROCEEDS (a permanently-unreachable node must not block
// reshard forever; the rejoin-during-catch-up residual is documented in the plan).
// A package-level var so tests can shorten it.
var reshardCutoverGateTimeout = 20 * time.Second

// reshardGateHook, when non-nil, is invoked at the START of awaitCutoverGate
// (after the Phase-4 cutover flip, while dual-write is still on and the old gen is
// still present) with the (collection, newGen) the gate was called with. It exists
// solely for tests to OBSERVE that the gate is reached with the correct arguments
// and to assert the cutover→gate→clear→drop ordering (e.g. that the old gen is
// still alive at the gate). It is nil in production. Set/reset under no concurrency
// in tests; reads are racy by design (single-coordinator reshard, test-only).
var reshardGateHook func(collection string, newGen uint32)

// awaitCutoverGate runs the Phase-4.5 all-nodes-applied cutover gate after the
// catalog flip and BEFORE the reshard clears dual-write / drops the old
// gen. It blocks (bounded by reshardCutoverGateTimeout) until every node
// in the cluster reports the new generation as live for collection — i.e. no node
// still routes a read to the old gen. Until it returns, dual-write is still on
// (Status==Resharding) and the old gen is still present + fresh, so a lagging
// follower routing to the old gen stays linearizable.
//
// It NEVER fails the reshard: on a gate timeout (some node unreachable / permanently
// lagging) it LOGS the unconfirmed nodes loudly and returns so the caller proceeds.
// Returning here leaves NO half-cutover inconsistency — the cutover flip
// already committed; this only delays retiring the old gen, and proceeding past a
// timeout is the documented best-effort behaviour (the alternative — hanging — is
// worse). Single-node / no-peers (or a nil node, defensive for non-cluster embed):
// the gate is a no-op (the local node IS the whole cluster), so it returns at once.
func (e *embedded) awaitCutoverGate(collection string, newGen uint32) {
	if reshardGateHook != nil {
		reshardGateHook(collection, newGen)
	}
	if e.node == nil {
		// No cluster node (defensive: the embedded path always has one, but a
		// single-node deployment has no meta-Raft / peers). The local node is the
		// whole cluster ⇒ nothing to wait for.
		return
	}
	if err := e.node.WaitAllNodesCatalogGen(collection, newGen, reshardCutoverGateTimeout); err != nil {
		// Best-effort gate: a node was unreachable / never confirmed within the
		// budget. Do NOT fail the reshard (an unreachable node must not block it
		// forever). Log loudly — the unconfirmed nodes are in the typed error — and
		// proceed to Phase 5/6. The residual (a node that rejoins and serves
		// catalog-routed reads during its own meta catch-up, before applying the
		// cutover, could briefly route to the now-dropped old gen) is accepted.
		slog.Warn("reshard cutover gate proceeding past timeout", "collection", collection, "gen", newGen, "err", err)
	}
}

// reshardCopyBatch is the number of records the Phase-3 copy processes between
// throttle pauses. 0 disables throttling (no pause). When >0 the copy sleeps
// reshardCopyPause after each batch so a reshard cannot saturate the cluster.
// Package vars so ops/tests can tune the copy rate.
var reshardCopyBatch = 0

// reshardCopyPause is the sleep inserted after each reshardCopyBatch records when
// throttling is enabled (reshardCopyBatch > 0). Ignored when reshardCopyBatch==0.
var reshardCopyPause = 0 * time.Millisecond

// VectorReshard repartitions a dense collection LIVE — reads AND writes stay up
// for the whole operation, no quiesce, no recreate. It runs the dual-write +
// background-copy state machine (Phases 0-6):
//
//	Phase 0  create-if-absent the new-gen (oldGen+1) physical partitions.
//	Phase 1  SetReshardState(Resharding) — turns on dual-write: every
//	         user point write now lands on BOTH the old (read) gen and the new gen.
//	Phase 2  drain grace — let any pre-Phase-1 (old-gen-only) write commit.
//	Phase 3  copy/reconcile: for each old-gen record, insert-IF-ABSENT into the
//	         new gen (Race A: a concurrent dual-write upsert always wins), then
//	         re-check the SOURCE old gen; if the id is gone, delete it from the new
//	         gen (Race B: never resurrect a concurrently-deleted id). A second full
//	         reconcile pass absorbs any dropped dual-write new-gen legs.
//	Phase 4  CUTOVER: SetPartitionsGen(newP,newGen) — atomic flip; reads now use
//	         the new gen. This is the single point of no return.
//	Phase 4.5 ALL-NODES-APPLIED GATE: waitAllNodesCatalogGen — block (bounded) until
//	         EVERY node routes to the new gen, with dual-write still on, so a lagging
//	         follower reads the still-fresh old gen. On timeout: log + proceed.
//	Phase 5  small settle grace, then SetReshardState(Stable) — writes go new-gen only.
//	Phase 6  drop the old-gen partitions (orphan-safe; VectorResplitCleanup is the
//	         backstop for any failed drop).
//
// Resumable: if the coordinator dies mid-reshard the collection is left
// Resharding (reads on old gen, dual-write on, status durable). Re-invoking
// VectorReshard with the SAME target re-runs Phase 0 idempotently (create-if-
// absent self-heals any partitions the interrupted run hadn't created), skips
// Phases 1 and 2 (state already durable, dual-write already on), then resumes
// from Phase 3; the copy is idempotent (if-absent + the resurrection guard make
// re-running whole partitions safe).
//
// Throttleable / bounded memory: the copy streams per old-partition (it does not
// buffer the whole collection) and supports a batch+pause throttle
// (reshardCopyBatch / reshardCopyPause) so it cannot saturate the cluster.
//
// Cost: dual-write doubles write cost for the reshard's duration only.
//
// The offline VectorResplit (quiesced bulk path) is unaffected — this is a new
// parallel capability.
func (e *embedded) VectorReshard(ctx context.Context, collection string, newP int) error {
	if newP <= 1 || newP > maxResplitPartitions {
		return fmt.Errorf("reshard: newP must be in [2, %d], got %d", maxResplitPartitions, newP)
	}
	// Crash-recovery FINALIZE (must precede newGen computation, resume-detect and
	// the newP==oldP no-op): if the coordinator died in the Phase-4→Phase-5 window
	// the cutover flip is durable but the Stable clear never ran, leaving the
	// collection permanently stuck Resharding toward its own live gen. Detect and
	// finalize idempotently before any normal flow.
	if err := e.finalizeIfPostCutover(ctx, collection, e.VectorResplitCleanup); err != nil {
		return fmt.Errorf("reshard: %w", err)
	}
	oldP, oldGen, ok := e.catalog.PartitionsGen(collection)
	if !ok || oldP <= 1 {
		return fmt.Errorf("reshard: %q is not partitioned", collection)
	}
	newGen := oldGen + 1

	// Resume detection: a collection already Resharding toward the SAME target
	// (newP/newGen) was interrupted before cutover — skip Phases 1-2 (already
	// applied: state durable, dual-write on) and resume from Phase 3. A
	// Resharding state toward a DIFFERENT target means a conflicting reshard is in
	// flight; refuse rather than clobber it. This MUST precede the newP==oldP
	// no-op check so a conflicting request during an active reshard surfaces an
	// error rather than being silently swallowed.
	resume := false
	if st, on := e.catalog.ReshardState(collection); on && st.Status == 1 {
		if st.NewP == newP && st.NewGen == newGen {
			resume = true
		} else {
			return fmt.Errorf("reshard: %q already resharding toward P=%d gen=%d (requested P=%d gen=%d); abort it first", collection, st.NewP, st.NewGen, newP, newGen)
		}
	}
	if newP == oldP && !resume {
		return nil // no-op (no reshard in flight toward this P)
	}

	// Create-if-absent the new-gen physical partitions. Read the config
	// from an old-gen physical partition (single-partition physical collections, so
	// Partitions=0). Self-heal: a prior interrupted attempt may have left a partial
	// set of new-gen partitions; create-if-absent makes Phase 0 idempotent for
	// resume (we DROP-then-create only on a fresh begin so a resumed run never
	// destroys already-copied data).
	cfgBody, err := e.Call(ctx, "vector_get_config", ops.EncodeGetConfigArgs(string(ops.PartitionKeyGen(collection, oldGen, 0))))
	if err != nil {
		return fmt.Errorf("reshard: get config: %w", err)
	}
	cfg, err := ops.DecodeGetConfigResult(cfgBody)
	if err != nil {
		return fmt.Errorf("reshard: decode config: %w", err)
	}
	cfg.Partitions = 0
	if !resume {
		// Fresh begin: drop any orphan new-gen partitions from an earlier failed
		// reshard so the create loop starts clean (mirrors VectorResplit's self-heal).
		for p := 0; p < newP; p++ {
			phys := string(ops.PartitionKeyGen(collection, newGen, p))
			if _, err := e.Call(ctx, "vector_drop_collection", ops.EncodeDropCollectionArgs(phys)); err != nil {
				return fmt.Errorf("reshard: pre-create cleanup of partition %d: %w", p, err)
			}
		}
	}
	for p := 0; p < newP; p++ {
		phys := string(ops.PartitionKeyGen(collection, newGen, p))
		// Create-if-absent: on resume the partition may already exist; tolerate that.
		if _, err := e.Call(ctx, "vector_create_collection", ops.EncodeCreateCollectionArgs(phys, cfg)); err != nil {
			if !resume {
				return fmt.Errorf("reshard: create new partition %d: %w", p, err)
			}
			// On resume, an "already exists" is expected; only a get-config failure
			// (partition genuinely absent AND create failed) is fatal.
			if _, gerr := e.Call(ctx, "vector_get_config", ops.EncodeGetConfigArgs(phys)); gerr != nil {
				return fmt.Errorf("reshard: resume self-heal create partition %d: %w", p, err)
			}
		}
	}

	// Turn on dual-write. Skipped on resume — the state is already
	// Resharding from the interrupted run, so re-applying would only be a
	// redundant Raft write. After this, every user insert/upsert/delete lands on
	// both gens.
	if !resume {
		if err := e.catalog.SetReshardState(collection, ReshardState{Status: 1, OldP: oldP, OldGen: oldGen, NewP: newP, NewGen: newGen}); err != nil {
			return fmt.Errorf("reshard: set state: %w", err)
		}
		// Drain grace so any write that began before dual-write was switched
		// on (old-gen-only) has committed and will be picked up by the copy. Skipped
		// on resume (dual-write was already on before the interruption).
		e.reshardSleep(ctx, reshardDrainGrace)
	}

	// Copy/reconcile. Two full passes: the first carries the bulk; the
	// second absorbs any dropped dual-write new-gen legs (a dual-write whose new
	// leg failed left the new gen missing a record). Both passes are idempotent.
	for pass := 0; pass < 2; pass++ {
		if err := e.reshardCopyPass(ctx, collection, oldP, oldGen, newP, newGen); err != nil {
			return err
		}
	}

	// CUTOVER — atomic catalog flip. Reads now route to the new gen.
	// Dual-write stays on (status still Resharding); since the reshard state pins
	// the OLD gen, dualTargets keeps writing BOTH gens AFTER the flip too (live=new,
	// target=old) instead of collapsing — so the old gen stays fresh for any node
	// still routing to it (the linearizable-catalog invariant). Point of no return.
	if err := e.catalog.SetPartitionsGen(collection, newP, newGen); err != nil {
		return fmt.Errorf("reshard: catalog flip: %w", err)
	}

	// ALL-NODES-APPLIED CUTOVER GATE. Block until EVERY node's local
	// catalog reports the new gen (or the gate times out + logs). Dual-write to BOTH
	// gens stays ON throughout (Status is still Resharding), so a meta-follower that
	// has not yet applied the flip routes its reads to the OLD gen — which is still
	// alive AND still receiving every write — and stays linearizable. Only once no
	// node still routes to the old gen is it safe to stop dual-write and drop it.
	// On timeout this proceeds (the residual is documented); it never hangs/fails.
	e.awaitCutoverGate(collection, newGen)

	// A SMALL settle grace for a request that was in-flight on a node at the
	// instant it applied the cutover (its read may have picked old-gen microseconds
	// before the flip while its dual-write leg still commits) — the gate gave the real
	// all-nodes-routed guarantee, this only drains those last cross-flip writes. Then
	// clear status — dual-write stops and writes target the new gen only.
	e.reshardSleep(ctx, reshardDrainGrace)
	if err := e.catalog.SetReshardState(collection, ReshardState{Status: 0}); err != nil {
		return fmt.Errorf("reshard: clear state: %w", err)
	}

	// Drop the old-gen partitions (orphan-safe — nothing routes there
	// post-flip; a failed drop just leaves orphans for VectorResplitCleanup). Reached
	// only AFTER the Phase-4.5 gate (or its timeout) confirmed no node routes old-gen.
	for p := 0; p < oldP; p++ {
		oldPhys := string(ops.PartitionKeyGen(collection, oldGen, p))
		if _, err := e.Call(ctx, "vector_drop_collection", ops.EncodeDropCollectionArgs(oldPhys)); err != nil {
			return fmt.Errorf("reshard: cutover done but dropping old partition %d failed: %w", p, err)
		}
	}
	return nil
}

// reshardCopyPass runs ONE full copy/reconcile pass over every old-gen partition.
// For each live record r in old-gen partition p it:
//
//  1. insert-IF-ABSENT into the new-gen partition PartitionOf(r.ID,newP). Atomic
//     single Raft op: a concurrent dual-write upsert of the same id always wins
//     (Race A) — if-absent never clobbers a live value.
//  2. resurrection guard (Race B): re-check r.ID's liveness on the SOURCE old-gen
//     partition; if it is GONE (deleted concurrently), delete r.ID from the
//     new-gen partition. This is the only thing preventing delete-resurrection and
//     is mandatory — see the plan's interleaving analysis.
//
// The scan is decoded per old-partition (bounded memory: one partition's records
// at a time, not the whole collection). Throttle: after every reshardCopyBatch
// records the pass sleeps reshardCopyPause (when reshardCopyBatch>0). Checkpointing
// is conceptually per old-partition: re-running a whole partition is idempotent
// (if-absent + the guard), so a resumed reshard re-runs partitions safely.
func (e *embedded) reshardCopyPass(ctx context.Context, collection string, oldP int, oldGen uint32, newP int, newGen uint32) error {
	processed := 0
	for p := 0; p < oldP; p++ {
		oldPhys := string(ops.PartitionKeyGen(collection, oldGen, p))
		body, err := e.Call(ctx, "vector_scan_vectors", ops.EncodeScanVectorsArgs(oldPhys))
		if err != nil {
			return fmt.Errorf("reshard: scan old partition %d: %w", p, err)
		}
		recs, err := ops.DecodeScanVectorsResult(body)
		if err != nil {
			return fmt.Errorf("reshard: decode scan %d: %w", p, err)
		}
		for _, r := range recs {
			newPhys := string(ops.PartitionKeyGen(collection, newGen, ops.PartitionOf(r.ID, newP)))
			var sparse vector.SparseVector
			if r.Sparse != nil {
				sparse = *r.Sparse
			}
			// (1) Atomic insert-if-absent — never clobbers a live dual-write (Race A).
			// Version-PRESERVING: carry the scanned point's exact per-point CAS version
			// (EncodeVectorInsertArgsVersionedKeyExpires) so an ONLINE reshard keeps
			// versions intact instead of resetting copied points to 1 — matching the
			// offline resplit backfill. A concurrent dual-write still wins (if-absent
			// no-op). r.KeyExpires carries the point's ABSOLUTE per-key payload deadlines
			// (from the scan trailer), set VERBATIM on a real insert (NOT recomputed) so
			// per-key TTLs survive the reshard time-stable; empty → byte-identical wire.
			if _, err := e.Call(ctx, "vector_insert_if_absent", ops.EncodeVectorInsertArgsVersionedKeyExpires(newPhys, r.ID, r.Vec, r.TTL, r.Metadata, sparse, r.Version, r.KeyExpires)); err != nil {
				return fmt.Errorf("reshard: copy id %d -> %s: %w", r.ID, newPhys, err)
			}
			// (2) Resurrection guard (Race B): if the id has since been deleted from the
			// SOURCE old gen, remove it from the new gen so the copy can't resurrect it.
			eb, err := e.Call(ctx, "vector_exists", ops.EncodeExistsArgs(oldPhys, r.ID))
			if err != nil {
				return fmt.Errorf("reshard: liveness probe id %d on %s: %w", r.ID, oldPhys, err)
			}
			live, err := ops.DecodeExistsResult(eb)
			if err != nil {
				return fmt.Errorf("reshard: decode liveness id %d: %w", r.ID, err)
			}
			if !live {
				if _, err := e.Call(ctx, "vector_delete", ops.EncodeVectorDeleteArgs(newPhys, r.ID)); err != nil {
					return fmt.Errorf("reshard: resurrection-guard delete id %d from %s: %w", r.ID, newPhys, err)
				}
			}
			processed++
			if reshardCopyBatch > 0 && processed%reshardCopyBatch == 0 && reshardCopyPause > 0 {
				e.reshardSleep(ctx, reshardCopyPause)
			}
		}
	}
	return nil
}

// reshardSleep sleeps for d, returning early if ctx is cancelled. It is the only
// pause primitive in the orchestrator (drain grace + copy throttle) — there are
// no busy loops.
func (e *embedded) reshardSleep(ctx context.Context, d time.Duration) {
	if d <= 0 {
		return
	}
	t := time.NewTimer(d)
	defer t.Stop()
	select {
	case <-ctx.Done():
	case <-t.C:
	}
}

// finalizeIfPostCutover recovers a collection stuck in the post-cutover /
// pre-Stable-clear crash window. If the coordinator dies between the Phase-4
// catalog flip (SetPartitionsGen(newP,newGen)) and the Phase-5 status clear
// (SetReshardState(Stable)), the durable state is: live PartitionsGen ==
// {st.NewP, st.NewGen} (cutover done) BUT status still Resharding toward that
// same target. Without recovery the collection is permanently un-reshardable: a
// re-invocation computes newGen=liveGen+1 (never matching st.NewGen) so resume
// never fires and it hits the conflicting-target error, while abort refuses a
// committed cutover.
//
// Detection condition (the crux): status is Resharding (Status==1) AND the live
// PartitionsGen already equals {st.NewP, st.NewGen}. This ONLY holds in the stuck
// post-cutover state — during a normal in-flight reshard the live gen is still the
// OLD gen (the flip has not happened), so live != target and finalize never fires.
//
// Finalize is idempotent: it runs Phase 5 (clear status -> Stable) then sweeps the
// stale non-live generations. Post-cutover the catalog no longer knows the true
// OLD P (st.OldP/OldGen are stale relative to the now-new live gen), so rather than
// dropping a known oldP×oldGen set we reuse the generation-sweep cleanup
// (VectorResplitCleanup / VectorMVResplitCleanup) which probes partition existence
// across every non-live generation and drops what it finds — exactly the orphan
// old gen. cleanup is the dense or MV cleanup passed by the caller.
func (e *embedded) finalizeIfPostCutover(ctx context.Context, collection string, cleanup func(context.Context, string) (int, error)) error {
	st, on := e.catalog.ReshardState(collection)
	if !on || st.Status != 1 {
		return nil // not resharding — nothing to finalize
	}
	liveP, liveGen, ok := e.catalog.PartitionsGen(collection)
	if !ok {
		return nil // not partitioned — leave the normal flow to report it
	}
	if liveP != st.NewP || liveGen != st.NewGen {
		return nil // pre-cutover (live still on old gen): a genuine in-flight reshard, not stuck
	}
	// Stuck post-cutover: Phase 5 (clear status) then sweep the orphan old gen.
	if err := e.catalog.SetReshardState(collection, ReshardState{Status: 0}); err != nil {
		return fmt.Errorf("finalize post-cutover: clear state: %w", err)
	}
	// Status is now Stable — the collection is no longer stuck. A cleanup failure
	// here only leaves orphan old-gen partitions (the VectorResplitCleanup backstop
	// sweeps them later); a retry of VectorReshard will already see Stable and skip
	// finalize, so this error is non-fatal to recoverability.
	if _, err := cleanup(ctx, collection); err != nil {
		return fmt.Errorf("finalize post-cutover: sweep stale gens (status already cleared; orphans left for cleanup): %w", err)
	}
	return nil
}

// VectorReshardAbort cancels an in-progress dense online reshard. It is valid
// ONLY before cutover — i.e. while the live PartitionsGen still reports the OLD
// gen (the Phase-4 flip has not happened). It clears the reshard status back to
// Stable on the old gen (turning off dual-write), then drops the new-gen
// partitions. Safe because reads never left the old gen and dual-writes to the
// old gen are the source of truth, so the collection is fully intact on the old
// gen. After cutover the reshard is committed; abort returns an error and the
// caller must run a NEW reshard to revert.
func (e *embedded) VectorReshardAbort(ctx context.Context, collection string) error {
	st, on := e.catalog.ReshardState(collection)
	if !on || st.Status != 1 {
		return fmt.Errorf("reshard abort: %q is not resharding", collection)
	}
	// Determine whether cutover already happened: after the flip the live
	// PartitionsGen reports the NEW gen/P, so live == target.
	liveP, liveGen, ok := e.catalog.PartitionsGen(collection)
	if !ok {
		return fmt.Errorf("reshard abort: %q is not partitioned", collection)
	}
	if liveGen == st.NewGen && liveP == st.NewP {
		return fmt.Errorf("reshard abort: %q reshard already cut over; re-invoke VectorReshard to finalize, then run a new reshard to revert", collection)
	}
	// Clear status -> Stable on the OLD (still-live) gen. Dual-write turns off.
	if err := e.catalog.SetReshardState(collection, ReshardState{Status: 0}); err != nil {
		return fmt.Errorf("reshard abort: clear state: %w", err)
	}
	// Drop the new-gen partitions (orphan-safe — nothing routes there after status
	// is cleared; a failed drop just leaves orphans for VectorResplitCleanup).
	for p := 0; p < st.NewP; p++ {
		phys := string(ops.PartitionKeyGen(collection, st.NewGen, p))
		if _, err := e.Call(ctx, "vector_drop_collection", ops.EncodeDropCollectionArgs(phys)); err != nil {
			return fmt.Errorf("reshard abort: dropping new-gen partition %d: %w", p, err)
		}
	}
	return nil
}

// VectorMVReshard is the multi-vector mirror of VectorReshard — it repartitions a
// multi-vector collection LIVE (reads AND writes stay up) using the identical
// dual-write + background-copy state machine (Phases 0-6), swapping the dense
// engine ops for their MV equivalents:
//
//	Phase 0  create-if-absent the new-gen (oldGen+1) physical MV partitions
//	         (vector_mv_create_collection; config via vector_mv_get_config).
//	Phase 1  SetReshardState(Resharding) — turns on dual-write: every
//	         VectorMVAdd / VectorMVDelete now lands on BOTH the old (read) gen and
//	         the new gen (routed via dualTargets, same helper as the dense path).
//	Phase 2  drain grace — let any pre-Phase-1 (old-gen-only) write commit.
//	Phase 3  copy/reconcile: for each old-gen doc, mv-add-IF-ABSENT into the new
//	         gen (Race A: a concurrent dual-write add always wins), then re-check the
//	         SOURCE old gen with mv-exists; if the docID is gone, mv-delete it from
//	         the new gen (Race B: never resurrect a concurrently-deleted doc). A
//	         second full reconcile pass absorbs any dropped dual-write new-gen legs.
//	Phase 4  CUTOVER: SetPartitionsGen(newP,newGen) — atomic flip; reads now use the
//	         new gen. Single point of no return.
//	Phase 5  drain grace, then SetReshardState(Stable) — writes go new-gen only.
//	Phase 6  drop the old-gen MV partitions (orphan-safe; VectorMVResplitCleanup is
//	         the backstop for any failed drop).
//
// The full token matrix AND metadata of every document are threaded through the
// copy (mv-add-if-absent carries Tokens + Metadata) so nothing is dropped across
// the reshard. Resumable and abortable exactly as the dense path (see VectorReshard
// / VectorReshardAbort). The offline VectorMVResplit (quiesced bulk path) is
// unaffected — this is a new parallel capability.
func (e *embedded) VectorMVReshard(ctx context.Context, collection string, newP int) error {
	if newP <= 1 || newP > maxResplitPartitions {
		return fmt.Errorf("mv reshard: newP must be in [2, %d], got %d", maxResplitPartitions, newP)
	}
	// Crash-recovery FINALIZE (mirror of the dense path; must precede newGen
	// computation, resume-detect and the newP==oldP no-op): finalize a collection
	// stuck post-cutover (flip durable, Stable clear never ran) before any normal flow.
	if err := e.finalizeIfPostCutover(ctx, collection, e.VectorMVResplitCleanup); err != nil {
		return fmt.Errorf("mv reshard: %w", err)
	}
	oldP, oldGen, ok := e.catalog.PartitionsGen(collection)
	if !ok || oldP <= 1 {
		return fmt.Errorf("mv reshard: %q is not partitioned", collection)
	}
	newGen := oldGen + 1

	// Resume detection (identical to dense): a collection already Resharding toward
	// the SAME target was interrupted before cutover — skip Phases 1-2 and resume
	// from Phase 3. A Resharding state toward a DIFFERENT target is a conflicting
	// reshard; refuse. This MUST precede the newP==oldP no-op check so a conflicting
	// request during an active reshard surfaces an error rather than being swallowed.
	resume := false
	if st, on := e.catalog.ReshardState(collection); on && st.Status == 1 {
		if st.NewP == newP && st.NewGen == newGen {
			resume = true
		} else {
			return fmt.Errorf("mv reshard: %q already resharding toward P=%d gen=%d (requested P=%d gen=%d); abort it first", collection, st.NewP, st.NewGen, newP, newGen)
		}
	}
	if newP == oldP && !resume {
		return nil // no-op (no reshard in flight toward this P)
	}

	// Create-if-absent the new-gen physical MV partitions. Config comes
	// from an old-gen physical partition (Partitions=0 — physical partitions are
	// single-partition). On a fresh begin we drop any orphan new-gen partitions
	// first (mirrors VectorMVResplit's self-heal); on resume we keep already-copied
	// data and only create the missing partitions.
	cfgBody, err := e.Call(ctx, "vector_mv_get_config", ops.EncodeMVGetConfigArgs(string(ops.PartitionKeyGen(collection, oldGen, 0))))
	if err != nil {
		return fmt.Errorf("mv reshard: get config: %w", err)
	}
	_, cfg, err := ops.DecodeMVCreateArgs(cfgBody)
	if err != nil {
		return fmt.Errorf("mv reshard: decode config: %w", err)
	}
	cfg.Partitions = 0
	if !resume {
		for p := 0; p < newP; p++ {
			phys := string(ops.PartitionKeyGen(collection, newGen, p))
			if _, err := e.Call(ctx, "vector_mv_drop_collection", ops.EncodeMVDeleteArgs(phys, 0)); err != nil {
				return fmt.Errorf("mv reshard: pre-create cleanup of partition %d: %w", p, err)
			}
		}
	}
	for p := 0; p < newP; p++ {
		phys := string(ops.PartitionKeyGen(collection, newGen, p))
		if _, err := e.Call(ctx, "vector_mv_create_collection", ops.EncodeMVCreateArgs(phys, cfg)); err != nil {
			if !resume {
				return fmt.Errorf("mv reshard: create new partition %d: %w", p, err)
			}
			// On resume an "already exists" is expected; only a genuinely-absent
			// partition (get-config also fails) is fatal.
			if _, gerr := e.Call(ctx, "vector_mv_get_config", ops.EncodeMVGetConfigArgs(phys)); gerr != nil {
				return fmt.Errorf("mv reshard: resume self-heal create partition %d: %w", p, err)
			}
		}
	}

	// Turn on dual-write. Skipped on resume (state already durable).
	if !resume {
		if err := e.catalog.SetReshardState(collection, ReshardState{Status: 1, OldP: oldP, OldGen: oldGen, NewP: newP, NewGen: newGen}); err != nil {
			return fmt.Errorf("mv reshard: set state: %w", err)
		}
		// Drain grace so any pre-dual-write (old-gen-only) write commits and
		// is picked up by the copy. Skipped on resume.
		e.reshardSleep(ctx, reshardDrainGrace)
	}

	// Copy/reconcile. Two full passes (the second absorbs dropped
	// dual-write new-gen legs); both idempotent.
	for pass := 0; pass < 2; pass++ {
		if err := e.mvReshardCopyPass(ctx, collection, oldP, oldGen, newP, newGen); err != nil {
			return err
		}
	}

	// CUTOVER — atomic catalog flip. Reads now route to the new gen.
	// Dual-write stays on (status still Resharding); because the reshard state pins
	// the OLD gen, dualTargets keeps writing BOTH gens AFTER the flip too — so the
	// old gen stays fresh for any node still routing to it (the linearizable-catalog
	// invariant, identical to the dense path). Point of no return.
	if err := e.catalog.SetPartitionsGen(collection, newP, newGen); err != nil {
		return fmt.Errorf("mv reshard: catalog flip: %w", err)
	}

	// ALL-NODES-APPLIED CUTOVER GATE (identical to the dense path). Block
	// until every node routes to the new gen (or timeout + log), with dual-write to
	// BOTH gens still ON, so a lagging follower reads the still-fresh old gen. Only
	// then is it safe to stop dual-write + drop the old gen. Never hangs/fails.
	e.awaitCutoverGate(collection, newGen)

	// A SMALL settle grace for a request in-flight on a node at the instant
	// it applied the cutover (the gate subsumes the old blind drain), then clear
	// status — writes now target the new gen only.
	e.reshardSleep(ctx, reshardDrainGrace)
	if err := e.catalog.SetReshardState(collection, ReshardState{Status: 0}); err != nil {
		return fmt.Errorf("mv reshard: clear state: %w", err)
	}

	// Drop the old-gen partitions (orphan-safe). Reached only AFTER the
	// Phase-4.5 gate (or its timeout) confirmed no node routes to the old gen.
	for p := 0; p < oldP; p++ {
		oldPhys := string(ops.PartitionKeyGen(collection, oldGen, p))
		if _, err := e.Call(ctx, "vector_mv_drop_collection", ops.EncodeMVDeleteArgs(oldPhys, 0)); err != nil {
			return fmt.Errorf("mv reshard: cutover done but dropping old partition %d failed: %w", p, err)
		}
	}
	return nil
}

// mvReshardCopyPass runs ONE full copy/reconcile pass over every old-gen MV
// partition — the multi-vector mirror of reshardCopyPass. For each live doc r in
// old-gen partition p it:
//
//  1. mv-add-IF-ABSENT into the new-gen partition PartitionOf(r.ID,newP), carrying
//     the FULL token matrix and metadata. Atomic single Raft op: a concurrent
//     dual-write mv-add of the same docID always wins (Race A) — if-absent never
//     clobbers a live value, and the metadata is never dropped.
//  2. resurrection guard (Race B): re-check r.ID's liveness on the SOURCE old-gen
//     partition with mv-exists; if it is GONE (deleted concurrently), mv-delete
//     r.ID from the new-gen partition. Mandatory — see the plan's interleaving
//     analysis.
//
// The scan is decoded per old-partition (bounded memory). Throttle and checkpoint
// semantics are identical to the dense pass (reshardCopyBatch / reshardCopyPause).
func (e *embedded) mvReshardCopyPass(ctx context.Context, collection string, oldP int, oldGen uint32, newP int, newGen uint32) error {
	processed := 0
	for p := 0; p < oldP; p++ {
		oldPhys := string(ops.PartitionKeyGen(collection, oldGen, p))
		body, err := e.Call(ctx, "vector_mv_scan_vectors", ops.EncodeMVScanArgs(oldPhys))
		if err != nil {
			return fmt.Errorf("mv reshard: scan old partition %d: %w", p, err)
		}
		recs, err := ops.DecodeMVScanResult(body)
		if err != nil {
			return fmt.Errorf("mv reshard: decode scan %d: %w", p, err)
		}
		for _, r := range recs {
			newPhys := string(ops.PartitionKeyGen(collection, newGen, ops.PartitionOf(r.ID, newP)))
			// (1) Atomic mv-add-if-absent — never clobbers a live dual-write (Race A);
			// carries the full token matrix + metadata. Version-preserving: the
			// versioned trailer makes the handler set the copied doc's per-document
			// CAS version VERBATIM (r.Version) instead of resetting to 1; version 0
			// ⇒ byte-identical to the plain if-absent wire (dual-write Race-A intact).
			// r.KeyExpires carries the doc's ABSOLUTE per-key payload deadlines (from
			// the scan trailer), set VERBATIM on a real add (NOT recomputed) so per-key
			// TTLs survive the reshard time-stable; empty → byte-identical wire.
			if _, err := e.Call(ctx, "vector_mv_add_if_absent", ops.EncodeMVAddArgsVersionedKeyExpiresSparse(newPhys, r.ID, r.Tokens, r.Metadata, r.Version, r.KeyExpires, r.Sparse)); err != nil {
				return fmt.Errorf("mv reshard: copy doc %d -> %s: %w", r.ID, newPhys, err)
			}
			// (2) Resurrection guard (Race B): if the doc has since been deleted from
			// the SOURCE old gen, remove it from the new gen.
			eb, err := e.Call(ctx, "vector_mv_exists", ops.EncodeMVExistsArgs(oldPhys, r.ID))
			if err != nil {
				return fmt.Errorf("mv reshard: liveness probe doc %d on %s: %w", r.ID, oldPhys, err)
			}
			live, err := ops.DecodeExistsResult(eb)
			if err != nil {
				return fmt.Errorf("mv reshard: decode liveness doc %d: %w", r.ID, err)
			}
			if !live {
				if _, err := e.Call(ctx, "vector_mv_delete", ops.EncodeMVDeleteArgs(newPhys, r.ID)); err != nil {
					return fmt.Errorf("mv reshard: resurrection-guard delete doc %d from %s: %w", r.ID, newPhys, err)
				}
			}
			processed++
			if reshardCopyBatch > 0 && processed%reshardCopyBatch == 0 && reshardCopyPause > 0 {
				e.reshardSleep(ctx, reshardCopyPause)
			}
		}
	}
	return nil
}

// VectorMVReshardAbort cancels an in-progress multi-vector online reshard — the MV
// mirror of VectorReshardAbort. Valid ONLY before cutover (while the live
// PartitionsGen still reports the OLD gen). It clears the reshard status back to
// Stable on the old gen (turning off dual-write), then drops the new-gen MV
// partitions. Safe because reads never left the old gen and dual-writes to it are
// the source of truth. After cutover the reshard is committed; abort errors (run a
// new reshard to revert).
func (e *embedded) VectorMVReshardAbort(ctx context.Context, collection string) error {
	st, on := e.catalog.ReshardState(collection)
	if !on || st.Status != 1 {
		return fmt.Errorf("mv reshard abort: %q is not resharding", collection)
	}
	liveP, liveGen, ok := e.catalog.PartitionsGen(collection)
	if !ok {
		return fmt.Errorf("mv reshard abort: %q is not partitioned", collection)
	}
	if liveGen == st.NewGen && liveP == st.NewP {
		return fmt.Errorf("mv reshard abort: %q reshard already cut over; re-invoke VectorMVReshard to finalize, then run a new reshard to revert", collection)
	}
	if err := e.catalog.SetReshardState(collection, ReshardState{Status: 0}); err != nil {
		return fmt.Errorf("mv reshard abort: clear state: %w", err)
	}
	for p := 0; p < st.NewP; p++ {
		phys := string(ops.PartitionKeyGen(collection, st.NewGen, p))
		if _, err := e.Call(ctx, "vector_mv_drop_collection", ops.EncodeMVDeleteArgs(phys, 0)); err != nil {
			return fmt.Errorf("mv reshard abort: dropping new-gen partition %d: %w", p, err)
		}
	}
	return nil
}

const (
	resplitCleanupGapTolerance = 256   // consecutive missing partitions before a generation is considered exhausted
	resplitCleanupMaxProbe     = 65536 // hard cap on partitions probed per generation
)

// maxResplitPartitions caps a resplit's target partition count as a sanity / abuse
// guard: a wrapped negative newP (int->uint32 over the wire) or an absurd value would
// otherwise drive billions of per-partition sub-ops. Far above any realistic partitioning.
const maxResplitPartitions = 65536

// VectorResplitCleanup drops every physical partition of the collection whose
// generation is not the current live generation — sweeping post-flip old-gen leaks
// and forward (pre-flip) orphans of failed resplits. The live generation is skipped
// so live data is never touched. Idempotent; returns the count dropped. Discovery is
// a bounded probe per non-live generation (the system has no collection enumeration).
func (e *embedded) VectorResplitCleanup(ctx context.Context, collection string) (int, error) {
	_, liveGen, ok := e.catalog.PartitionsGen(collection)
	if !ok {
		return 0, fmt.Errorf("resplit cleanup: %q is not partitioned", collection)
	}
	dropped := 0
	for g := uint32(0); g <= liveGen+1; g++ {
		if g == liveGen {
			continue
		}
		n, err := e.dropGenerationPartitions(ctx, collection, g)
		if err != nil {
			return dropped, err
		}
		dropped += n
	}
	return dropped, nil
}

// dropGenerationPartitions drops every existing physical partition of (collection, gen),
// probing p=0.. and dropping the ones that exist, stopping after
// resplitCleanupGapTolerance consecutive absent partitions (or the hard cap). Returns
// the count dropped.
func (e *embedded) dropGenerationPartitions(ctx context.Context, collection string, gen uint32) (int, error) {
	dropped, miss := 0, 0
	for p := 0; p < resplitCleanupMaxProbe && miss < resplitCleanupGapTolerance; p++ {
		phys := string(ops.PartitionKeyGen(collection, gen, p))
		if _, err := e.Call(ctx, "vector_get_config", ops.EncodeGetConfigArgs(phys)); err != nil {
			miss++
			continue
		}
		miss = 0
		if _, err := e.Call(ctx, "vector_drop_collection", ops.EncodeDropCollectionArgs(phys)); err != nil {
			return dropped, fmt.Errorf("resplit cleanup: drop %s: %w", phys, err)
		}
		dropped++
	}
	return dropped, nil
}

// VectorMVResplit performs the offline generational repartition for a multi-vector
// collection, mirroring VectorResplit EXACTLY (swapping the dense ops for their MV
// equivalents). Cutover ordering is the correctness crux: the new generation's
// partitions are FULLY BUILT before the catalog flip (so all reads/point-ops route
// to the intact old generation until then), the flip is ONE atomic catalog write,
// and the old generation is dropped LAST (orphan-safe — nothing routes there
// post-flip). If resplit fails before the flip, the collection is fully intact (the
// new-gen partitions are orphans, a documented retry/cleanup concern). OFFLINE: the
// caller MUST quiesce writes first.
func (e *embedded) VectorMVResplit(ctx context.Context, collection string, newP int) error {
	if newP <= 1 || newP > maxResplitPartitions {
		return fmt.Errorf("mv resplit: newP must be in [2, %d], got %d", maxResplitPartitions, newP)
	}
	oldP, oldGen, ok := e.catalog.PartitionsGen(collection)
	if !ok || oldP <= 1 {
		return fmt.Errorf("mv resplit: %q is not partitioned", collection)
	}
	if newP == oldP {
		return nil // no-op
	}
	newGen := oldGen + 1
	// 1. Read existing config from an old-gen physical partition.
	cfgBody, err := e.Call(ctx, "vector_mv_get_config", ops.EncodeMVGetConfigArgs(string(ops.PartitionKeyGen(collection, oldGen, 0))))
	if err != nil {
		return fmt.Errorf("mv resplit: get config: %w", err)
	}
	_, cfg, err := ops.DecodeMVCreateArgs(cfgBody)
	if err != nil {
		return fmt.Errorf("mv resplit: decode config: %w", err)
	}
	cfg.Partitions = 0 // physical partitions are single-partition
	// Self-heal: a prior resplit attempt that failed before the catalog flip may
	// have left newGen partitions behind. Drop them first (no-op if absent) so the
	// create loop below starts clean and a retry succeeds.
	for p := 0; p < newP; p++ {
		phys := string(ops.PartitionKeyGen(collection, newGen, p))
		if _, err := e.Call(ctx, "vector_mv_drop_collection", ops.EncodeMVDeleteArgs(phys, 0)); err != nil {
			return fmt.Errorf("mv resplit: pre-create cleanup of partition %d: %w", p, err)
		}
	}
	// 2. Create the new generation's physical partitions.
	for p := 0; p < newP; p++ {
		phys := string(ops.PartitionKeyGen(collection, newGen, p))
		if _, err := e.Call(ctx, "vector_mv_create_collection", ops.EncodeMVCreateArgs(phys, cfg)); err != nil {
			return fmt.Errorf("mv resplit: create new partition %d: %w", p, err)
		}
	}
	// 3. Stream every document old gen -> new gen, re-hashed by newP.
	//
	// COMMIT BATCHING + CONCURRENT BUILD: the naive copy did one Raft-committed
	// vector_mv_add_versioned per document, so MV resplit was commit- AND
	// build-bound. MV resplit is OFFLINE (quiesced), and the new-gen partitions are
	// fresh (empty), so we ACCUMULATE every document by its target new partition
	// across ALL old-partition scans, then copy each partition's whole document set
	// with ONE vector_mv_add_batch op. Into an empty partition that op takes the
	// concurrent bulk-build path (MultiBulkBuild → one multi-core inner-graph build),
	// so each new partition costs ONE Raft commit and ONE concurrent build instead of
	// one commit + serial token inserts per document. The applied result is
	// equivalent (same per-doc version/metadata/sparse/TTL; the inner graph is built
	// concurrently, like the dense bulk path).
	byPart := make([][]vector.MultiScanRecord, newP)
	for p := 0; p < oldP; p++ {
		oldPhys := string(ops.PartitionKeyGen(collection, oldGen, p))
		body, err := e.Call(ctx, "vector_mv_scan_vectors", ops.EncodeMVScanArgs(oldPhys))
		if err != nil {
			return fmt.Errorf("mv resplit: scan old partition %d: %w", p, err)
		}
		recs, err := ops.DecodeMVScanResult(body)
		if err != nil {
			return fmt.Errorf("mv resplit: decode scan %d: %w", p, err)
		}
		for _, r := range recs {
			np := ops.PartitionOf(r.ID, newP)
			byPart[np] = append(byPart[np], r)
		}
	}
	for np := 0; np < newP; np++ {
		if len(byPart[np]) == 0 {
			continue
		}
		newPhys := string(ops.PartitionKeyGen(collection, newGen, np))
		if _, err := e.Call(ctx, "vector_mv_add_batch", ops.EncodeMVAddBatchArgs(newPhys, byPart[np])); err != nil {
			return fmt.Errorf("mv resplit: batch-add partition %d: %w", np, err)
		}
	}
	// 4. Atomic cutover: flip catalog to {newP, newGen}. After this, all routing
	// uses the new generation.
	if err := e.catalog.SetPartitionsGen(collection, newP, newGen); err != nil {
		return fmt.Errorf("mv resplit: catalog flip: %w", err)
	}
	// 5. Drop the old generation's partitions (orphan-safe: nothing routes there
	// post-flip).
	for p := 0; p < oldP; p++ {
		oldPhys := string(ops.PartitionKeyGen(collection, oldGen, p))
		if _, err := e.Call(ctx, "vector_mv_drop_collection", ops.EncodeMVDeleteArgs(oldPhys, 0)); err != nil {
			return fmt.Errorf("mv resplit: cutover done but dropping old partition %d failed: %w", p, err)
		}
	}
	return nil
}

// VectorMVResplitCleanup drops every physical partition of the multi-vector
// collection whose generation is not the current live generation — sweeping
// post-flip old-gen leaks and forward (pre-flip) orphans of failed resplits. The
// live generation is skipped so live data is never touched. Idempotent; returns the
// count dropped. Discovery is a bounded probe per non-live generation (the system
// has no collection enumeration). Mirrors VectorResplitCleanup, reusing the same
// probe bounds.
func (e *embedded) VectorMVResplitCleanup(ctx context.Context, collection string) (int, error) {
	_, liveGen, ok := e.catalog.PartitionsGen(collection)
	if !ok {
		return 0, fmt.Errorf("mv resplit cleanup: %q is not partitioned", collection)
	}
	dropped := 0
	for g := uint32(0); g <= liveGen+1; g++ {
		if g == liveGen {
			continue
		}
		n, err := e.dropMVGenerationPartitions(ctx, collection, g)
		if err != nil {
			return dropped, err
		}
		dropped += n
	}
	return dropped, nil
}

// dropMVGenerationPartitions drops every existing physical MV partition of
// (collection, gen), probing p=0.. and dropping the ones that exist, stopping after
// resplitCleanupGapTolerance consecutive absent partitions (or the hard cap).
// Returns the count dropped. Mirrors dropGenerationPartitions.
func (e *embedded) dropMVGenerationPartitions(ctx context.Context, collection string, gen uint32) (int, error) {
	dropped, miss := 0, 0
	for p := 0; p < resplitCleanupMaxProbe && miss < resplitCleanupGapTolerance; p++ {
		phys := string(ops.PartitionKeyGen(collection, gen, p))
		if _, err := e.Call(ctx, "vector_mv_get_config", ops.EncodeMVGetConfigArgs(phys)); err != nil {
			miss++
			continue
		}
		miss = 0
		if _, err := e.Call(ctx, "vector_mv_drop_collection", ops.EncodeMVDeleteArgs(phys, 0)); err != nil {
			return dropped, fmt.Errorf("mv resplit cleanup: drop %s: %w", phys, err)
		}
		dropped++
	}
	return dropped, nil
}
