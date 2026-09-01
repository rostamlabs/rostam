// SPDX-License-Identifier: Apache-2.0

package cluster

import (
	"bytes"
	"encoding/gob"
	"errors"
)

// ReshardEntry is one collection's online-reshard state in the meta catalog.
// Status 0 = Stable (steady), 1 = Resharding (a live repartition is dual-writing
// to the target gen). TargetP/TargetGen are the new gen being copied into;
// SourceP/SourceGen pin the OLD gen at reshard-begin so the dual-write keeps
// hitting the old gen even AFTER the cutover flips the live catalog to the new gen
// (a lagging follower routing to the old gen then still finds it fresh). The zero
// value is Stable, so a nil CatalogReshard map (old gob snapshots) reads as every
// collection Stable. SourceP/SourceGen are added like TargetP/TargetGen were: gob
// matches by name, so an OLD entry without them decodes Source=0 — a reshard begun
// pre-upgrade then degrades to the old collapse-at-cutover behavior (acceptable;
// new reshards carry Source).
type ReshardEntry struct {
	Status    uint8
	TargetP   int
	TargetGen uint32
	SourceP   int
	SourceGen uint32
}

// ShardEpochPair is one (shardID, epoch) claim carried in an OpShardLeaseRenew
// liveness beacon (automatic failover). A beaconing node batches one pair
// per shard it currently primaries, so a single beacon op carries the node's whole
// primary set — bounding Raft traffic to N_nodes entries per interval rather than
// N_shards. The FSM applies each pair epoch/primary-guarded and mutates NO
// replicated state (it only fires the leader-local liveness observer); see
// MetaFSM.Apply's OpShardLeaseRenew case.
type ShardEpochPair struct {
	ShardID int
	Epoch   uint64
}

// AliasAction is one mutation in an atomic alias batch (OpSetAliasBatch). All
// names are canonical. Delete=true removes the alias entry; otherwise the alias
// is created/overwritten to point at Canonical. A batch of N actions applies in
// one FSM-lock region (one Raft log entry), so a swap such as
// {delete prod, create prod→v2} never exposes an intermediate state.
type AliasAction struct {
	Alias     string // canonical alias name (the key)
	Canonical string // canonical target collection (ignored when Delete)
	Delete    bool
}

// State is the meta-Raft FSM payload. Always replace-by-overwrite — no
// per-field deltas later.
type State struct {
	NumShards int
	Members   []Peer     // canonical member list, sorted by NodeID
	Placement [][]string // Placement[shardID] = NodeIDs hosting that shard. In 5b: always == NodeIDs(Members).
	// Catalog maps a canonical collection name (e.g. "default/docs") to its
	// partition count P (>1). Absent / nil means single-partition. This is the
	// cluster-wide durable partition catalog read by every node for routing.
	Catalog map[string]uint32
	// CatalogGen is a PARALLEL map (keyed identically to Catalog) carrying each
	// collection's partition generation, bumped on offline resplit. Kept separate
	// from Catalog so old gob snapshots (written before generations existed) still
	// decode: the missing field deserializes to nil, i.e. every collection at
	// generation 0. nil / absent entry means generation 0.
	CatalogGen map[string]uint32
	// CatalogReshard is a PARALLEL map (keyed identically to Catalog) carrying each
	// collection's online-reshard state. Kept separate from Catalog/CatalogGen so
	// old gob snapshots (written before resharding existed) still decode: the
	// missing field deserializes to nil, i.e. every collection Stable. A nil map or
	// an absent / zero-Status entry means Stable.
	CatalogReshard map[string]ReshardEntry
	// Aliases is a PARALLEL map (alias-canonical name → target-canonical
	// collection name) carrying the collection-alias catalog. Kept separate from
	// Catalog/CatalogGen/CatalogReshard so old gob snapshots (written before
	// aliases existed) still decode: the missing field deserializes to nil, i.e.
	// no aliases. A nil map or an absent entry means the name is not an alias.
	Aliases map[string]string
	// ShardEpoch, ShardPrimary, ShardISR are three PARALLEL maps (all keyed by
	// shardID) holding the primary-backup / ISR replication CONTROL PLANE (see
	// shard/pbisr/DESIGN.md). They are ADDITIVE and INERT — nothing
	// consumes them yet (the data plane arrives later) — so they must not
	// change any existing behavior. Kept separate from Placement so old gob
	// snapshots (written before ISR replication existed) still decode: each
	// missing field deserializes to nil, i.e. no shard has a leadership epoch,
	// primary, or ISR set. gob matches by field name, so this is pure back-compat.
	//
	// ShardEpoch maps shardID → current leadership epoch. Monotonic; bumped by
	// MetaRaft on each primary election. A nil map or an absent entry reads as
	// epoch 0 (no leadership generation established yet).
	ShardEpoch map[int]uint64
	// ShardPrimary maps shardID → the nodeID that holds the shard's current
	// epoch. A nil map or an absent entry reads as "" (no primary).
	ShardPrimary map[int]string
	// ShardFormer maps shardID → the nodeID designated to BOOTSTRAP that shard's
	// Raft group at initial cluster formation. Kept separate from Placement so old
	// gob snapshots decode with it nil (no shard has a designated former).
	//
	// This exists because a shard's Raft group has to be CREATED by exactly one of
	// its owners — hashicorp raft's BootstrapCluster — and the node-level
	// `-bootstrap` flag is the wrong authority for that decision. A node hosts only
	// the shards it owns, so when ReplicationFactor < len(members) some shards have
	// an owner set that EXCLUDES the bootstrap node; every owner then ran with
	// Bootstrap=false, nobody called BootstrapCluster, and those groups stayed
	// configuration-less and leaderless forever (writes to any key hashing there
	// hung). Recording the designated former in the meta log makes formation a
	// CONTROL-PLANE decision that any owner can act on, independent of which node
	// carried the operator's flag.
	//
	// Entries are WRITE-ONCE: MetaFSM.Apply ignores an OpSetShardFormer for a shard
	// that already has one. That is what keeps this safe against the case the
	// per-node alternative could not handle — a fresh-disk node rejoining an
	// ESTABLISHED cluster finds the shard already formed, so it never bootstraps a
	// rival group and instead joins as a follower and replays from the leader.
	ShardFormer map[int]string
	// ShardISR maps shardID → the in-sync replica nodeIDs for the current epoch
	// (includes the primary). A nil map or an absent entry reads as nil (no ISR
	// set). A new epoch resets this to just {primary} until backups catch up.
	ShardISR map[int][]string
	// MinISR is the STRUCTURAL durability floor for OpSetShardISR: the FSM refuses
	// to commit an ISR set smaller than this, so a buggy shrink/grow driver can
	// never drop a shard below the floor at the FSM level (it is a backstop under
	// the driver's own decidePBShrink floor). Seeded at bootstrap from Config.MinISR
	// (carried in the bootstrap OpSetMembers entry). A zero value (an old gob
	// snapshot, or raft mode) DISABLES the floor — back-compat: gob decodes a
	// missing field as 0, and the OpSetShardISR floor check treats 0 as "no floor"
	// (only the len==0 guard still applies). This is cluster-wide, not per-shard;
	// a per-shard floor is a later refinement. Guarded by the FSM lock like the
	// rest of State. NEVER blocks the election reset (that is OpSetShardEpoch, which
	// deliberately resets the ISR to {primary} — a different op, not floor-checked).
	MinISR               int
	ReplicationFactor    int
	ReplicationFactorSet bool
	// LastIndex carries the MetaFSM command frontier (its applied-command index)
	// INTO a snapshot so a snapshot-restored follower does not under-report 0 and
	// wait forever for the readIndex barrier. It is FSM-applied metadata, NOT
	// catalog content: it is stamped onto the snapshot copy in MetaFSM.Snapshot and
	// read back in MetaFSM.Restore; it is NOT populated by Apply and is deliberately
	// excluded from State() equality (equalState) and the catalog deep-copy. gob
	// matches by field name, so old snapshots (no LastIndex) decode it as 0 — the
	// tracker then catches up on the next applied command (the node replays the log
	// after restore). Mirrors how CatalogGen/CatalogReshard/Aliases were added.
	LastIndex uint64
}

func encodeState(s State) ([]byte, error) {
	var buf bytes.Buffer
	if err := gob.NewEncoder(&buf).Encode(s); err != nil {
		return nil, err
	}
	return buf.Bytes(), nil
}

func decodeState(b []byte) (State, error) {
	if len(b) == 0 {
		return State{}, errors.New("cluster: empty state")
	}
	var s State
	if err := gob.NewDecoder(bytes.NewReader(b)).Decode(&s); err != nil {
		return State{}, err
	}
	return s, nil
}

// Op tags meta-Raft log entries.
type Op uint8

// Op constants for meta-Raft log entries.
const (
	OpUnknown    Op = 0 // zero value; never written deliberately
	OpSetMembers Op = 1
	// reserved: OpAddMember = 2, OpRemoveMember = 3 for future phases
	OpSetPlacement      Op = 4  // online rebalancing: set one shard's owner set
	OpSetCatalogEntry   Op = 5  // set one collection's partition count in the catalog
	OpSetCatalogReshard Op = 6  // set one collection's online-reshard state
	OpSetAliasBatch     Op = 7  // atomically apply a batch of alias mutations
	OpSetShardEpoch     Op = 8  // primary-backup/ISR: bump one shard's leadership epoch + primary (+ ISR when it is a seed)
	OpSetShardISR       Op = 9  // primary-backup/ISR: set one shard's in-sync replica set
	OpShardLeaseRenew   Op = 10 // primary-backup failover: a node's batched primary-liveness beacon (inert in the FSM)
	OpSetShardFormer    Op = 11 // cluster formation: designate the one owner that bootstraps a shard's Raft group
)

// logBody is the gob payload of a LogEntry (everything after the 1-byte op
// tag). gob matches by field name, so adding fields stays backward compatible:
// entries written before a field existed decode it as the zero value.
type logBody struct {
	Members           []Peer
	NumShards         int
	ReplicationFactor int
	ShardID           int
	Owners            []string
	Collection        string           // OpSetCatalogEntry / OpSetCatalogReshard: canonical collection name
	Partitions        uint32           // OpSetCatalogEntry: partition count P
	Generation        uint32           // OpSetCatalogEntry: partition generation (0 until resplit)
	ReshardStatus     uint8            // OpSetCatalogReshard: 0=Stable, 1=Resharding
	ReshardTargetP    int              // OpSetCatalogReshard: target partition count
	ReshardTargetGen  uint32           // OpSetCatalogReshard: target generation
	ReshardSourceP    int              // OpSetCatalogReshard: source (old) partition count
	ReshardSourceGen  uint32           // OpSetCatalogReshard: source (old) generation
	AliasBatch        []AliasAction    // OpSetAliasBatch: alias mutations applied atomically
	Epoch             uint64           // OpSetShardEpoch / OpSetShardISR: shard leadership epoch (ShardID is the key)
	Primary           string           // OpSetShardEpoch: nodeID that holds this epoch
	ISR               []string         // OpSetShardISR (and OpSetShardEpoch when it is a SEED): in-sync replica nodeIDs for this epoch
	Node              string           // OpShardLeaseRenew: the beaconing node claiming the LeaseRenew primaries
	LeaseRenew        []ShardEpochPair // OpShardLeaseRenew: (shard,epoch) pairs the node currently primaries
	MinISR            int              // OpSetMembers: cluster-wide structural ISR floor (seeded at bootstrap)
}

// LogEntry is the wire format of a meta-Raft log entry: 1-byte op tag
// + gob-encoded payload.
type LogEntry struct {
	Op                Op
	Members           []Peer
	NumShards         int
	ReplicationFactor int // 0 = full replication (every member hosts every shard)
	// OpSetPlacement fields:
	ShardID int
	Owners  []string
	// OpSetCatalogEntry fields:
	Collection string // canonical collection name (e.g. "default/docs")
	Partitions uint32 // partition count P (>1)
	Generation uint32 // partition generation (0 until resplit)
	// OpSetCatalogReshard fields (Collection above is reused as the key):
	ReshardStatus    uint8  // 0=Stable, 1=Resharding
	ReshardTargetP   int    // target partition count
	ReshardTargetGen uint32 // target generation
	ReshardSourceP   int    // source (old) partition count, pinned at reshard-begin
	ReshardSourceGen uint32 // source (old) generation, pinned at reshard-begin
	// OpSetAliasBatch field: the batch of alias mutations applied atomically.
	AliasBatch []AliasAction
	// primary-backup/ISR control-plane fields (ShardID above is reused as the key):
	Epoch   uint64   // OpSetShardEpoch / OpSetShardISR: shard leadership epoch
	Primary string   // OpSetShardEpoch: nodeID that holds this epoch
	ISR     []string // OpSetShardISR: in-sync replica nodeIDs for this epoch. Also set on
	// OpSetShardEpoch by a SEED (ApplySetShardSeed) so bootstrap/restore commit
	// (epoch, primary, ISR) ATOMICALLY; empty on a failover bump, which resets
	// the ISR to {primary} (see the OpSetShardEpoch apply).
	// OpShardLeaseRenew (primary-backup failover liveness beacon) fields:
	Node       string           // the beaconing node claiming the LeaseRenew primaries
	LeaseRenew []ShardEpochPair // (shard,epoch) pairs the node currently primaries
	// OpSetMembers: cluster-wide structural ISR floor seeded into MetaState at bootstrap.
	MinISR int
}

func encodeLogEntry(e LogEntry) ([]byte, error) {
	var buf bytes.Buffer
	if err := buf.WriteByte(byte(e.Op)); err != nil {
		return nil, err
	}
	if err := gob.NewEncoder(&buf).Encode(logBody{
		Members:           e.Members,
		NumShards:         e.NumShards,
		ReplicationFactor: e.ReplicationFactor,
		ShardID:           e.ShardID,
		Owners:            e.Owners,
		Collection:        e.Collection,
		Partitions:        e.Partitions,
		Generation:        e.Generation,
		ReshardStatus:     e.ReshardStatus,
		ReshardTargetP:    e.ReshardTargetP,
		ReshardTargetGen:  e.ReshardTargetGen,
		ReshardSourceP:    e.ReshardSourceP,
		ReshardSourceGen:  e.ReshardSourceGen,
		AliasBatch:        e.AliasBatch,
		Epoch:             e.Epoch,
		Primary:           e.Primary,
		ISR:               e.ISR,
		Node:              e.Node,
		LeaseRenew:        e.LeaseRenew,
		MinISR:            e.MinISR,
	}); err != nil {
		return nil, err
	}
	return buf.Bytes(), nil
}

func decodeLogEntry(b []byte) (LogEntry, error) {
	if len(b) < 1 {
		return LogEntry{}, errors.New("cluster: log entry too short")
	}
	e := LogEntry{Op: Op(b[0])}
	var body logBody
	if err := gob.NewDecoder(bytes.NewReader(b[1:])).Decode(&body); err != nil {
		return LogEntry{}, err
	}
	e.Members = body.Members
	e.NumShards = body.NumShards
	e.ReplicationFactor = body.ReplicationFactor
	e.ShardID = body.ShardID
	e.Owners = body.Owners
	e.Collection = body.Collection
	e.Partitions = body.Partitions
	e.Generation = body.Generation
	e.ReshardStatus = body.ReshardStatus
	e.ReshardTargetP = body.ReshardTargetP
	e.ReshardTargetGen = body.ReshardTargetGen
	e.ReshardSourceP = body.ReshardSourceP
	e.ReshardSourceGen = body.ReshardSourceGen
	e.AliasBatch = body.AliasBatch
	e.Epoch = body.Epoch
	e.Primary = body.Primary
	e.ISR = body.ISR
	e.Node = body.Node
	e.LeaseRenew = body.LeaseRenew
	e.MinISR = body.MinISR
	return e, nil
}
