// SPDX-License-Identifier: Apache-2.0

package wire

import "encoding/binary"

// Read-consistency levels carried in a read op's opts trailer. The values mirror
// the cluster.Consistency enum (AnyReplica=0, LeaderOnly=1) so the byte threads
// wire-compatibly from the API edge through fan-out to the shard. Linearizable=2
// is opt-in and additive: AnyReplica/LeaderOnly bytes are unchanged.
const (
	ConsistencyAnyReplica       uint8 = 0 // load-balanced replica read (default)
	ConsistencyLeaderOnly       uint8 = 1 // read from the partition's Raft leader (best-effort)
	ConsistencyLinearizable     uint8 = 2 // readIndex barrier (VerifyLeader + commit-index catch-up)
	ConsistencyBoundedStaleness uint8 = 3 // any-replica read with a max-staleness bound (8-byte raft-entry bound rides the trailer)
)

// ReadOptsTrailerMarker is the self-delimiting marker byte that, when present at
// the end of a get / get_config arg blob, signals an [rc:u8][opa:u8] opts block
// follows. It mirrors NamedTrailerOpts (the named-search trailer): a zero marker
// is NEVER emitted, so a legacy (no-trailer) blob — or one whose rc/opa are both
// zero — is byte-identical to the pre-rc form. Value 1<<0 matches NamedTrailerOpts
// so the wire convention is uniform across the read families.
const ReadOptsTrailerMarker uint8 = 1 << 0 // [rc:u8][opa:u8] follow

// ReadOptsStalenessBit is the SECOND marker bit, OR'd into the trailer marker ONLY
// for a ConsistencyBoundedStaleness read. When set, an 8-byte big-endian staleness
// bound (max raft entries the served replica may lag the leader's committed
// frontier) follows the [marker][rc][opa] block. It is additive: rc∈{0,1,2} never
// set it, so those trailers are byte-identical to the pre-bounded-staleness form.
const ReadOptsStalenessBit uint8 = 1 << 1 // an 8-byte big-endian bound follows [marker][rc][opa]

// AppendReadOptsTrailerBounded appends the self-delimiting opts trailer to base,
// optionally carrying an 8-byte big-endian staleness bound. It is the single source
// of truth the legacy AppendReadOptsTrailer wraps:
//   - rc==0 && opa==0 → base UNCHANGED (byte-identical to the legacy / no-trailer
//     form), so the AnyReplica default path is wire-identical and the bound is never
//     on the wire when unused.
//   - rc==ConsistencyBoundedStaleness → marker = ReadOptsTrailerMarker|ReadOptsStalenessBit,
//     then [marker][rc][opa] followed by the 8 bound bytes (big-endian).
//   - any other rc/opa (rc∈{1,2}, or rc==0 with opa!=0) → [marker][rc][opa], EXACTLY
//     the current 3-byte form ⇒ byte-identical to the pre-bounded-staleness trailer.
//
// The bound rides ONLY when rc==ConsistencyBoundedStaleness, regardless of its
// value (a bound==0 bounded-staleness read still carries the 8 zero bytes so the
// staleness bit, and hence the level, survives a round-trip unambiguously).
func AppendReadOptsTrailerBounded(base []byte, rc, opa uint8, bound uint64) []byte {
	if rc == 0 && opa == 0 {
		return base // byte-identical to the legacy / no-trailer form
	}
	if rc == ConsistencyBoundedStaleness {
		out := append(base, ReadOptsTrailerMarker|ReadOptsStalenessBit, rc, opa)
		var b [8]byte
		binary.BigEndian.PutUint64(b[:], bound)
		return append(out, b[:]...)
	}
	return append(base, ReadOptsTrailerMarker, rc, opa)
}

// appendBoundTail appends the 8-byte big-endian staleness bound to out, but ONLY
// when rc==ConsistencyBoundedStaleness. For every other rc it returns out
// unchanged, so the wire is byte-identical to the pre-bounded-staleness form for
// rc∈{0,1,2}. This is the shared raw-tail helper used by the flag-bit and
// presence-byte read ops (dense search, hybrid, groups, MV/named search, named
// scroll, …) whose rc/opa already ride a custom marker — they call this directly
// after emitting [rc][opa] so the bound rides immediately behind, decoded
// symmetrically by readBoundTail.
func appendBoundTail(out []byte, rc uint8, bound uint64) []byte {
	if rc != ConsistencyBoundedStaleness {
		return out
	}
	var b [8]byte
	binary.BigEndian.PutUint64(b[:], bound)
	return append(out, b[:]...)
}

// readBoundTail reads the 8-byte big-endian staleness bound that appendBoundTail
// emits, but ONLY when rc==ConsistencyBoundedStaleness. For other rc it returns
// (0, off, nil) leaving off untouched. A set bound (rc==3) with fewer than 8
// trailing bytes is corruption — fail loud (never a silent drop, so a bounded
// read never silently degrades its freshness SLO).
func readBoundTail(args []byte, off int, rc uint8) (bound uint64, newOff int, err error) {
	if rc != ConsistencyBoundedStaleness {
		return 0, off, nil
	}
	if len(args) < off+8 {
		return 0, off, ErrVectorArgsTruncated
	}
	return binary.BigEndian.Uint64(args[off : off+8]), off + 8, nil
}

// AppendReadOptsTrailer is the legacy 3-byte trailer codec, now a thin wrapper over
// AppendReadOptsTrailerBounded with bound==0. Existing callers (rc∈{0,1,2}) are
// byte-identical to before — they never set the staleness bit (which only fires for
// rc==ConsistencyBoundedStaleness). ZERO churn.
func AppendReadOptsTrailer(base []byte, rc, opa uint8) []byte {
	return AppendReadOptsTrailerBounded(base, rc, opa, 0)
}

// DecodeReadOptsTrailerBounded reads the optional [marker][rc][opa](+[bound:u64])
// trailer that may sit at offset n of args (the bytes consumed by the base decode).
// Backward-compatible: no trailer (or a zero marker, never emitted ⇒ treated as "no
// trailer" for trailing-byte tolerance) yields rc=0, opa=0, bound=0. A present
// marker with a truncated rc/opa block, OR a set staleness bit with a truncated
// 8-byte bound, is corruption — fail loud (never a silent drop, so a bounded /
// linearizable read can never silently degrade to stale).
func DecodeReadOptsTrailerBounded(args []byte, n int) (rc, opa uint8, bound uint64, err error) {
	if len(args) <= n || args[n] == 0 {
		return 0, 0, 0, nil
	}
	marker := args[n]
	off := n + 1
	if marker&ReadOptsTrailerMarker != 0 {
		if len(args) < off+2 {
			return 0, 0, 0, ErrVectorArgsTruncated
		}
		rc = args[off]
		opa = args[off+1]
		off += 2
	}
	if marker&ReadOptsStalenessBit != 0 {
		if len(args) < off+8 {
			return 0, 0, 0, ErrVectorArgsTruncated
		}
		bound = binary.BigEndian.Uint64(args[off : off+8])
	}
	return rc, opa, bound, nil
}

// DecodeReadOptsTrailer reads the optional [marker][rc][opa] trailer, delegating to
// DecodeReadOptsTrailerBounded and dropping the bound so existing callers are
// unchanged. A bounded-staleness trailer's 8 bound bytes are still consumed/validated
// (fail-loud) by the delegate; this view simply does not surface them.
func DecodeReadOptsTrailer(args []byte, n int) (rc, opa uint8, err error) {
	rc, opa, _, err = DecodeReadOptsTrailerBounded(args, n)
	return rc, opa, err
}

// ReadStalenessOf peeks the max-staleness bound out of a ConsistencyBoundedStaleness
// read op's args. It is OP-AWARE — a switch IDENTICAL in shape to ReadConsistencyOf,
// delegating to each op's real Decode*Opts (which now also return the bound) — so it
// is correct (never a blind fixed offset) and panic-safe (bounds-checked). It runs on
// the OpReadOnly serve path for bounded-staleness reads only, so it is cheap.
//
// It returns (bound, true) ONLY when the op is a recognized read op whose decoded
// rc==ConsistencyBoundedStaleness; every other op (writes, admin, non-bounded reads)
// returns (0, false) ⇒ the shard guard treats the bound as 0 (strictest: serve only a
// fully-caught-up replica), which fails safe rather than serving over-stale data.
func ReadStalenessOf(op string, args []byte) (bound uint64, ok bool) {
	switch op {
	case "vector_search", "vector_search_docs":
		_, _, _, _, rc, _, b, err := DecodeVectorSearchArgsOpts(args)
		if err != nil || rc != ConsistencyBoundedStaleness {
			return 0, false
		}
		return b, true
	case "vector_hybrid_search", "vector_hybrid_lanes":
		_, _, _, _, _, rc, _, b, err := DecodeHybridSearchArgsOpts(args)
		if err != nil || rc != ConsistencyBoundedStaleness {
			return 0, false
		}
		return b, true
	case "vector_search_groups", "vector_group_candidates":
		_, _, _, _, rc, _, b, err := DecodeGroupSearchArgsOpts(args)
		if err != nil || rc != ConsistencyBoundedStaleness {
			return 0, false
		}
		return b, true
	case "vector_search_text":
		_, _, _, _, rc, _, b, err := DecodeSearchTextArgsOpts(args)
		if err != nil || rc != ConsistencyBoundedStaleness {
			return 0, false
		}
		return b, true
	case "vector_hybrid_text", "vector_hybrid_text_lanes":
		_, _, _, _, _, rc, _, b, err := DecodeHybridTextArgsOpts(args)
		if err != nil || rc != ConsistencyBoundedStaleness {
			return 0, false
		}
		return b, true
	case "vector_bm25_stats":
		_, _, rc, _, b, err := DecodeBM25StatsArgs(args)
		if err != nil || rc != ConsistencyBoundedStaleness {
			return 0, false
		}
		return b, true
	case "vector_scroll":
		_, _, _, rc, _, _, _, b, err := DecodeScrollArgsCursor(args)
		if err != nil || rc != ConsistencyBoundedStaleness {
			return 0, false
		}
		return b, true
	case "vector_mv_search":
		_, _, _, _, rc, _, b, err := DecodeMVSearchArgsOpts(args)
		if err != nil || rc != ConsistencyBoundedStaleness {
			return 0, false
		}
		return b, true
	case "vector_named_search", "vector_named_search_docs":
		_, _, _, _, _, rc, _, b, err := DecodeNamedSearchArgsOpts(args)
		if err != nil || rc != ConsistencyBoundedStaleness {
			return 0, false
		}
		return b, true
	case "vector_named_sparse_search":
		_, _, _, _, _, rc, _, b, err := DecodeNamedSparseSearchArgsOpts(args)
		if err != nil || rc != ConsistencyBoundedStaleness {
			return 0, false
		}
		return b, true
	case "vector_named_hybrid_search", "vector_named_hybrid_lanes":
		_, _, _, _, _, _, _, rc, _, b, err := DecodeNamedHybridArgs(args)
		if err != nil || rc != ConsistencyBoundedStaleness {
			return 0, false
		}
		return b, true
	case "vector_mv_hybrid_search", "vector_mv_hybrid_lanes":
		_, _, _, _, _, rc, _, b, err := DecodeMVHybridArgs(args)
		if err != nil || rc != ConsistencyBoundedStaleness {
			return 0, false
		}
		return b, true
	case "vector_named_scroll":
		_, _, _, _, _, rc, _, b, err := DecodeNamedScrollArgsOpts(args)
		if err != nil || rc != ConsistencyBoundedStaleness {
			return 0, false
		}
		return b, true
	case "vector_mv_scroll":
		_, _, _, rc, _, _, _, b, err := DecodeMVScrollArgsOpts(args)
		if err != nil || rc != ConsistencyBoundedStaleness {
			return 0, false
		}
		return b, true
	case "vector_query", "vector_named_query", "vector_mv_query":
		_, _, rc, _, b, err := DecodeQueryArgs(args)
		if err != nil || rc != ConsistencyBoundedStaleness {
			return 0, false
		}
		return b, true
	case "vector_get", "vector_named_get", "vector_mv_get":
		_, _, _, rc, _, b, err := DecodeVectorGetArgsOpts(args)
		if err != nil || rc != ConsistencyBoundedStaleness {
			return 0, false
		}
		return b, true
	case "vector_get_config":
		_, rc, _, b, err := DecodeGetConfigArgsOpts(args)
		if err != nil || rc != ConsistencyBoundedStaleness {
			return 0, false
		}
		return b, true
	case "vector_named_get_config":
		_, rc, _, b, err := DecodeNamedNameArgsOpts(args)
		if err != nil || rc != ConsistencyBoundedStaleness {
			return 0, false
		}
		return b, true
	case "vector_mv_get_config":
		_, rc, _, b, err := DecodeMVGetConfigArgsOpts(args)
		if err != nil || rc != ConsistencyBoundedStaleness {
			return 0, false
		}
		return b, true
	default:
		return 0, false
	}
}

// ReadConsistencyOf peeks the read_consistency byte out of a read op's args
// WITHOUT a full / allocating decode being load-bearing — it is OP-AWARE and runs
// on every OpReadOnly serve, so it must be O(1)-ish and panic-safe. It returns
// (rc, true) only for the read ops whose wire layout carries a consistency opts
// trailer; every other op (writes, admin, and read ops with no consistency byte)
// returns (0, false) ⇒ the caller treats it as AnyReplica with no barrier.
//
// Which ops carry the byte (verified against each op's encoder, single source of
// truth in ops/vector.go / ops/multivector.go):
//
//   - vector_search / vector_search_docs → EncodeVectorSearchArgsOpts:
//     [rc][opa] appended behind the vecFlagSearchOpts bit in args[0].
//   - vector_hybrid_search             → EncodeHybridSearchArgsOpts: [rc][opa] tail.
//   - vector_hybrid_lanes              → EncodeHybridSearchArgsOpts (same wire as
//     vector_hybrid_search): the partitioned hybrid fan-out's per-partition op.
//   - vector_search_groups             → EncodeGroupSearchArgsOpts: [present=1][rc][opa] tail.
//   - vector_group_candidates          → EncodeGroupSearchArgsOpts (same wire as
//     vector_search_groups): the partitioned group fan-out's per-partition op.
//   - vector_scroll                    → EncodeScrollArgsCursor:    [present=1][rc][opa][...] tail.
//   - vector_mv_search                 → EncodeMVSearchArgsOpts:    [present=1][rc][opa] tail.
//   - vector_named_search / _search_docs → EncodeNamedSearchArgsOpts: [marker][rc][opa]
//     tail behind a self-delimiting marker bit (both ops share the named-search wire).
//   - vector_named_scroll              → EncodeNamedScrollArgsOpts: [marker][?afterID][rc][opa]
//     tail (the cursor + opts share one marker bitfield).
//   - vector_mv_scroll                 → EncodeMVScrollArgsOpts:    [marker][?afterID][rc][opa]
//     tail (the cursor + opts share one marker bitfield; mirrors vector_named_scroll).
//   - vector_get / vector_named_get / vector_mv_get → EncodeVectorGetArgsOpts:
//     [marker][rc][opa] self-delimiting tail behind the fixed-length get base
//     block (all three get families share the get wire).
//   - vector_get_config / vector_named_get_config / vector_mv_get_config →
//     Encode{GetConfig,NamedName,MVGetConfig}ArgsOpts: [marker][rc][opa] tail
//     behind the single-name base block (the meta-barrier arms in embedded;
//     the shard data barrier arms here for the catalog read).
//
// Because each layout differs (a flag bit vs a self-delimiting presence byte,
// variable-length query/filter blocks in between), peeking a fixed offset would be
// fragile — a wrong offset would mis-classify a read as Linearizable (a perf hit)
// or, worse, serve a Linearizable read WITHOUT the barrier (a correctness hole).
// So this delegates to each op's REAL bounds-checked Decode* function, which never
// panics and yields rc=0 on legacy (no-trailer) args. The cost is the same decode
// the handler does moments later; correctness is guaranteed by reuse, not by a
// hand-maintained offset table.
//
// The named family (vector_named_search / _search_docs / _scroll) now carries the
// read_consistency byte via the marker-bitfield opts trailer added by
// EncodeNamedSearchArgsOpts / EncodeNamedScrollArgsOpts, so it is handled below
// exactly like the dense/MV families — arming the shard data barrier for a
// Linearizable named read (without these cases the barrier never fires and the
// read silently degrades to stale).
func ReadConsistencyOf(op string, args []byte) (rc uint8, ok bool) {
	switch op {
	case "vector_search", "vector_search_docs":
		_, _, _, _, rc, _, _, err := DecodeVectorSearchArgsOpts(args)
		if err != nil {
			return 0, false
		}
		return rc, true
	case "vector_hybrid_search", "vector_hybrid_lanes":
		// vector_hybrid_lanes is the per-partition op the hybrid fan-out scatters;
		// it shares vector_hybrid_search's wire layout (its handler decodes with
		// DecodeHybridSearchArgs), so the same Opts decoder reads its rc trailer.
		// Without this case a forwarded/served Linearizable hybrid read on a
		// partitioned collection would skip the barrier ⇒ stale-capable.
		_, _, _, _, _, rc, _, _, err := DecodeHybridSearchArgsOpts(args)
		if err != nil {
			return 0, false
		}
		return rc, true
	case "vector_search_groups", "vector_group_candidates":
		// vector_group_candidates is the per-partition op the group fan-out
		// scatters; it shares vector_search_groups' wire layout (its handler
		// decodes with DecodeGroupSearchArgs), so the same Opts decoder reads its
		// rc trailer. Without this case a forwarded/served Linearizable group read
		// on a partitioned collection would skip the barrier ⇒ stale-capable.
		_, _, _, _, rc, _, _, err := DecodeGroupSearchArgsOpts(args)
		if err != nil {
			return 0, false
		}
		return rc, true
	case "vector_search_text":
		// The BM25 full-text search carries rc/opa behind textFlagOpts; without this
		// case a Linearizable text read skips the shard barrier ⇒ stale-capable.
		_, _, _, _, rc, _, _, err := DecodeSearchTextArgsOpts(args)
		if err != nil {
			return 0, false
		}
		return rc, true
	case "vector_hybrid_text", "vector_hybrid_text_lanes":
		// vector_hybrid_text_lanes is the per-partition op the text fan-out scatters; it
		// shares vector_hybrid_text's wire (decoded with DecodeHybridTextArgs), so one
		// Opts decoder reads rc for both. Without this case a Linearizable hybrid-text
		// read on a partitioned collection skips the barrier ⇒ stale-capable.
		_, _, _, _, _, rc, _, _, err := DecodeHybridTextArgsOpts(args)
		if err != nil {
			return 0, false
		}
		return rc, true
	case "vector_bm25_stats":
		// Phase 0 of the global-DF text fan-out carries rc/opa behind a self-delimiting
		// marker trailer; without this case a Linearizable phase-0 stats read skips the
		// shard barrier ⇒ the gathered df/n could be stale.
		_, _, rc, _, _, err := DecodeBM25StatsArgs(args)
		if err != nil {
			return 0, false
		}
		return rc, true
	case "vector_scroll":
		_, _, _, rc, _, _, _, _, err := DecodeScrollArgsCursor(args)
		if err != nil {
			return 0, false
		}
		return rc, true
	case "vector_mv_search":
		_, _, _, _, rc, _, _, err := DecodeMVSearchArgsOpts(args)
		if err != nil {
			return 0, false
		}
		return rc, true
	case "vector_named_search", "vector_named_search_docs":
		// Both share the named-search wire (handleNamedSearch / handleNamedSearchDocs
		// decode with DecodeNamedSearchArgs), so one Opts decoder reads the rc
		// trailer for both. Without these cases a Linearizable named read skips the
		// shard barrier ⇒ stale-capable.
		_, _, _, _, _, rc, _, _, err := DecodeNamedSearchArgsOpts(args)
		if err != nil {
			return 0, false
		}
		return rc, true
	case "vector_named_sparse_search":
		// The sparse-lane named search carries rc/opa behind the same self-delimiting
		// marker as vector_named_search; without this case a Linearizable named sparse
		// read skips the shard barrier ⇒ stale-capable.
		_, _, _, _, _, rc, _, _, err := DecodeNamedSparseSearchArgsOpts(args)
		if err != nil {
			return 0, false
		}
		return rc, true
	case "vector_named_hybrid_search", "vector_named_hybrid_lanes":
		// The cross-space named hybrid (fused search + the fan-out lanes leaf) carries
		// rc/opa in the EncodeNamedHybridArgs flags trailer; without this case a
		// Linearizable named hybrid skips the shard barrier ⇒ stale-capable. Both ops
		// share the codec, so one decoder reads rc for both.
		_, _, _, _, _, _, _, rc, _, _, err := DecodeNamedHybridArgs(args)
		if err != nil {
			return 0, false
		}
		return rc, true
	case "vector_mv_hybrid_search", "vector_mv_hybrid_lanes":
		// The MV cross-modality hybrid (fused search + the fan-out lanes leaf) carries
		// rc/opa in the EncodeMVHybridArgs flags trailer; without this case a
		// Linearizable MV hybrid skips the shard barrier ⇒ stale-capable. Both ops share
		// the codec, so one decoder reads rc for both.
		_, _, _, _, _, rc, _, _, err := DecodeMVHybridArgs(args)
		if err != nil {
			return 0, false
		}
		return rc, true
	case "vector_named_scroll":
		_, _, _, _, _, rc, _, _, err := DecodeNamedScrollArgsOpts(args)
		if err != nil {
			return 0, false
		}
		return rc, true
	case "vector_mv_scroll":
		_, _, _, rc, _, _, _, _, err := DecodeMVScrollArgsOpts(args)
		if err != nil {
			return 0, false
		}
		return rc, true
	case "vector_query", "vector_named_query", "vector_mv_query":
		// The unified Query API ops (dense + named + MV) carry rc/opa in the shared
		// self-delimiting read-opts trailer (EncodeQueryArgs →
		// AppendReadOptsTrailerBounded); without this case a Linearizable query
		// skips the shard barrier ⇒ stale-capable.
		_, _, rc, _, _, err := DecodeQueryArgs(args)
		if err != nil {
			return 0, false
		}
		return rc, true
	case "vector_get", "vector_named_get", "vector_mv_get":
		// All three get families share the EncodeVectorGetArgs / ...Opts wire
		// (the get arg shape is identical), so one Opts decoder reads the rc
		// trailer for all three. Without these cases a Linearizable point-get
		// skips the shard barrier ⇒ stale-capable (a point-get could serve a
		// just-overwritten / just-deleted point's old state).
		_, _, _, rc, _, _, err := DecodeVectorGetArgsOpts(args)
		if err != nil {
			return 0, false
		}
		return rc, true
	case "vector_get_config":
		_, rc, _, _, err := DecodeGetConfigArgsOpts(args)
		if err != nil {
			return 0, false
		}
		return rc, true
	case "vector_named_get_config":
		_, rc, _, _, err := DecodeNamedNameArgsOpts(args)
		if err != nil {
			return 0, false
		}
		return rc, true
	case "vector_mv_get_config":
		_, rc, _, _, err := DecodeMVGetConfigArgsOpts(args)
		if err != nil {
			return 0, false
		}
		return rc, true
	case "kv_query":
		// kv_query carries its read_consistency byte inline, not behind a
		// trailer marker (see DecodeKVQueryArgs's layout), so a full decode
		// is the only way to reach it — the same cost the handler pays
		// moments later, matching every other case here. A kv_query can
		// never be ConsistencyBoundedStaleness (DecodeKVQueryArgs rejects
		// it outright, no bound trailer in this frame), so there is no
		// corresponding ReadStalenessOf case to add.
		a, err := DecodeKVQueryArgs(args)
		if err != nil {
			return 0, false
		}
		return a.Consistency, true
	default:
		return 0, false
	}
}
