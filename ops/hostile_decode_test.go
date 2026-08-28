// SPDX-License-Identifier: Apache-2.0

package ops

import (
	"encoding/binary"
	"reflect"
	"testing"
)

// NO DECODER MAY PANIC ON HOSTILE BYTES.
//
// Every op arrives as an untrusted []byte from the network and is decoded
// BEFORE much else happens to it. A decoder that panics is therefore not a
// rejected request but a dead process, and on the replicated path a write is
// committed to the log before it is decoded — so a frame that panics one node
// panics every node that applies it.
//
// The specific hazard this sweep exists for is 32-bit. Lengths are read with
// int(binary.BigEndian.Uint32(...)), which is exact on 64-bit but widens
// NEGATIVE on 32-bit for anything above MaxInt32 — and a negative length
// SATISFIES the `len(args) < off+n` checks written to reject it, so the decode
// proceeds into a slice bound or a negative make(). There are ~180 such
// conversions in this package. Auditing them by hand is laborious and, worse,
// unverifiable; calling every decoder with lengths chosen to trigger the
// widening is neither, and under GOARCH=386 (the 32-bit CI lane) it exercises
// the conversion where it actually misbehaves.
//
// The bar is deliberately low and absolute: return whatever you like, in any
// combination, but do not panic. That makes the sweep cheap to extend — a new
// decoder is one line — and it cannot go stale into a false pass, because a
// decoder that stops rejecting hostile input still has to survive it.
//
// Reflection rather than 140 hand-written wrappers: every one of these takes a
// single []byte and they differ only in what they return, which is exactly the
// shape reflect.Call handles uniformly. Hand-wrapping them would have meant 140
// arities to get right and to keep right.

var hostileDecoders = []struct {
	name string
	fn   any
}{
	{"DecodeAliasBatchArgs", DecodeAliasBatchArgs},
	{"DecodeAliasListArgs", DecodeAliasListArgs},
	{"DecodeAliasListResult", DecodeAliasListResult},
	{"DecodeBM25StatsArgs", DecodeBM25StatsArgs},
	{"DecodeBM25StatsResult", DecodeBM25StatsResult},
	{"DecodeBulkBuildArgs", DecodeBulkBuildArgs},
	{"DecodeBulkStageArgs", DecodeBulkStageArgs},
	{"DecodeBulkStagePayloadArgs", DecodeBulkStagePayloadArgs},
	{"DecodeCADArgs", DecodeCADArgs},
	{"DecodeCASArgs", DecodeCASArgs},
	{"DecodeClearPayloadArgs", DecodeClearPayloadArgs},
	{"DecodeClearPayloadArgsCAS", DecodeClearPayloadArgsCAS},
	{"DecodeCreateCollectionArgs", DecodeCreateCollectionArgs},
	{"DecodeDeleteByFilterArgs", DecodeDeleteByFilterArgs},
	{"DecodeDeleteByFilterResult", DecodeDeleteByFilterResult},
	{"DecodeDeletePayloadKeysArgs", DecodeDeletePayloadKeysArgs},
	{"DecodeDeletePayloadKeysArgsCAS", DecodeDeletePayloadKeysArgsCAS},
	{"DecodeDropCollectionArgs", DecodeDropCollectionArgs},
	{"DecodeExistsArgs", DecodeExistsArgs},
	{"DecodeExistsResult", DecodeExistsResult},
	{"DecodeGetConfigArgs", DecodeGetConfigArgs},
	{"DecodeGetConfigArgsOpts", DecodeGetConfigArgsOpts},
	{"DecodeGetConfigResult", DecodeGetConfigResult},
	{"DecodeGroupSearchArgs", DecodeGroupSearchArgs},
	{"DecodeGroupSearchArgsOpts", DecodeGroupSearchArgsOpts},
	{"DecodeGroups", DecodeGroups},
	{"DecodeGroupsDegraded", DecodeGroupsDegraded},
	{"DecodeGroupsDegradedRaw", DecodeGroupsDegradedRaw},
	{"DecodeHybridLanesResult", DecodeHybridLanesResult},
	{"DecodeHybridResults", DecodeHybridResults},
	{"DecodeHybridResultsDegraded", DecodeHybridResultsDegraded},
	{"DecodeHybridSearchArgs", DecodeHybridSearchArgs},
	{"DecodeHybridSearchArgsOpts", DecodeHybridSearchArgsOpts},
	{"DecodeHybridTextArgs", DecodeHybridTextArgs},
	{"DecodeHybridTextArgsGlobal", DecodeHybridTextArgsGlobal},
	{"DecodeHybridTextArgsOpts", DecodeHybridTextArgsOpts},
	{"DecodeIfAbsentResult", DecodeIfAbsentResult},
	{"DecodeIncrResult", DecodeIncrResult},
	{"DecodeKeysAddArgs", DecodeKeysAddArgs},
	{"DecodeKeysListResult", DecodeKeysListResult},
	{"DecodeKeysRevokeArgs", DecodeKeysRevokeArgs},
	{"DecodeMVAddArgs", DecodeMVAddArgs},
	{"DecodeMVAddArgsCAS", DecodeMVAddArgsCAS},
	{"DecodeMVAddArgsCASKeyTTL", DecodeMVAddArgsCASKeyTTL},
	{"DecodeMVAddArgsCASKeyTTLSparse", DecodeMVAddArgsCASKeyTTLSparse},
	{"DecodeMVAddArgsVersioned", DecodeMVAddArgsVersioned},
	{"DecodeMVAddArgsVersionedKeyExpires", DecodeMVAddArgsVersionedKeyExpires},
	{"DecodeMVAddArgsVersionedKeyExpiresSparse", DecodeMVAddArgsVersionedKeyExpiresSparse},
	{"DecodeMVAddBatchArgs", DecodeMVAddBatchArgs},
	{"DecodeMVCreateArgs", DecodeMVCreateArgs},
	{"DecodeMVDeleteArgs", DecodeMVDeleteArgs},
	{"DecodeMVDeleteArgsCAS", DecodeMVDeleteArgsCAS},
	{"DecodeMVExistsArgs", DecodeMVExistsArgs},
	{"DecodeMVGetBatchResult", DecodeMVGetBatchResult},
	{"DecodeMVGetConfigArgs", DecodeMVGetConfigArgs},
	{"DecodeMVGetConfigArgsOpts", DecodeMVGetConfigArgsOpts},
	{"DecodeMVGetResult", DecodeMVGetResult},
	{"DecodeMVGetResultV", DecodeMVGetResultV},
	{"DecodeMVHybridArgs", DecodeMVHybridArgs},
	{"DecodeMVResults", DecodeMVResults},
	{"DecodeMVResultsDegraded", DecodeMVResultsDegraded},
	{"DecodeMVScanArgs", DecodeMVScanArgs},
	{"DecodeMVScanResult", DecodeMVScanResult},
	{"DecodeMVScrollArgsOpts", DecodeMVScrollArgsOpts},
	{"DecodeMVScrollArgsOrder", DecodeMVScrollArgsOrder},
	{"DecodeMVSearchArgs", DecodeMVSearchArgs},
	{"DecodeMVSearchArgsOpts", DecodeMVSearchArgsOpts},
	{"DecodeMVSearchArgsOptsFilter", DecodeMVSearchArgsOptsFilter},
	{"DecodeNamedConfigResult", DecodeNamedConfigResult},
	{"DecodeNamedCreateArgs", DecodeNamedCreateArgs},
	{"DecodeNamedDeleteArgs", DecodeNamedDeleteArgs},
	{"DecodeNamedDeleteArgsCAS", DecodeNamedDeleteArgsCAS},
	{"DecodeNamedGetBatchResult", DecodeNamedGetBatchResult},
	{"DecodeNamedGetResult", DecodeNamedGetResult},
	{"DecodeNamedGetResultV", DecodeNamedGetResultV},
	{"DecodeNamedHybridArgs", DecodeNamedHybridArgs},
	{"DecodeNamedInsertArgs", DecodeNamedInsertArgs},
	{"DecodeNamedInsertArgsCAS", DecodeNamedInsertArgsCAS},
	{"DecodeNamedInsertArgsKeyTTL", DecodeNamedInsertArgsKeyTTL},
	{"DecodeNamedInsertArgsSparseKeyTTL", DecodeNamedInsertArgsSparseKeyTTL},
	{"DecodeNamedNameArgs", DecodeNamedNameArgs},
	{"DecodeNamedNameArgsOpts", DecodeNamedNameArgsOpts},
	{"DecodeNamedScrollArgs", DecodeNamedScrollArgs},
	{"DecodeNamedScrollArgsCursor", DecodeNamedScrollArgsCursor},
	{"DecodeNamedScrollArgsOpts", DecodeNamedScrollArgsOpts},
	{"DecodeNamedScrollArgsOrder", DecodeNamedScrollArgsOrder},
	{"DecodeNamedSearchArgs", DecodeNamedSearchArgs},
	{"DecodeNamedSearchArgsOpts", DecodeNamedSearchArgsOpts},
	{"DecodeNamedSparseSearchArgs", DecodeNamedSparseSearchArgs},
	{"DecodeNamedSparseSearchArgsOpts", DecodeNamedSparseSearchArgsOpts},
	{"DecodePayloadResult", DecodePayloadResult},
	{"DecodePutBatchArgs", DecodePutBatchArgs},
	{"DecodePutBatchResult", DecodePutBatchResult},
	{"DecodeQueryArgs", DecodeQueryArgs},
	{"DecodeQueryResult", DecodeQueryResult},
	{"DecodeQueryResultDegraded", DecodeQueryResultDegraded},
	{"DecodeQueryResultGroupedFanOut", DecodeQueryResultGroupedFanOut},
	{"DecodeQuerySpecArgs", DecodeQuerySpecArgs},
	{"DecodeQueryTreeLanes", DecodeQueryTreeLanes},
	{"DecodeReshardAbortArgs", DecodeReshardAbortArgs},
	{"DecodeReshardArgs", DecodeReshardArgs},
	{"DecodeResplitArgs", DecodeResplitArgs},
	{"DecodeResplitCleanupArgs", DecodeResplitCleanupArgs},
	{"DecodeResplitCleanupResult", DecodeResplitCleanupResult},
	{"DecodeScanVectorsArgs", DecodeScanVectorsArgs},
	{"DecodeScanVectorsResult", DecodeScanVectorsResult},
	{"DecodeScrollArgs", DecodeScrollArgs},
	{"DecodeScrollArgsCursor", DecodeScrollArgsCursor},
	{"DecodeScrollArgsOpts", DecodeScrollArgsOpts},
	{"DecodeScrollArgsOrder", DecodeScrollArgsOrder},
	{"DecodeScrollResult", DecodeScrollResult},
	{"DecodeScrollResultRaw", DecodeScrollResultRaw},
	{"DecodeSearchTextArgs", DecodeSearchTextArgs},
	{"DecodeSearchTextArgsGlobal", DecodeSearchTextArgsGlobal},
	{"DecodeSearchTextArgsOpts", DecodeSearchTextArgsOpts},
	{"DecodeSetPayloadArgs", DecodeSetPayloadArgs},
	{"DecodeSetPayloadArgsCAS", DecodeSetPayloadArgsCAS},
	{"DecodeSetPayloadArgsOpts", DecodeSetPayloadArgsOpts},
	{"DecodeTopology", DecodeTopology},
	{"DecodeVectorDeleteArgs", DecodeVectorDeleteArgs},
	{"DecodeVectorDeleteArgsCAS", DecodeVectorDeleteArgsCAS},
	{"DecodeVectorDocs", DecodeVectorDocs},
	{"DecodeVectorDocsDegraded", DecodeVectorDocsDegraded},
	{"DecodeVectorDocsDegradedRaw", DecodeVectorDocsDegradedRaw},
	{"DecodeVectorDocsRaw", DecodeVectorDocsRaw},
	{"DecodeVectorGetArgs", DecodeVectorGetArgs},
	{"DecodeVectorGetArgsOpts", DecodeVectorGetArgsOpts},
	{"DecodeVectorGetBatchArgs", DecodeVectorGetBatchArgs},
	{"DecodeVectorGetBatchResult", DecodeVectorGetBatchResult},
	{"DecodeVectorGetResult", DecodeVectorGetResult},
	{"DecodeVectorGetResultV", DecodeVectorGetResultV},
	{"DecodeVectorInsertArgs", DecodeVectorInsertArgs},
	{"DecodeVectorInsertArgsCAS", DecodeVectorInsertArgsCAS},
	{"DecodeVectorInsertArgsKeyExpires", DecodeVectorInsertArgsKeyExpires},
	{"DecodeVectorInsertArgsKeyTTL", DecodeVectorInsertArgsKeyTTL},
	{"DecodeVectorSearchArgs", DecodeVectorSearchArgs},
	{"DecodeVectorSearchArgsOpts", DecodeVectorSearchArgsOpts},
	{"DecodeVectorSearchResults", DecodeVectorSearchResults},
	{"DecodeVectorSearchResultsDegraded", DecodeVectorSearchResultsDegraded},
	{"DecodeWASMRegistration", DecodeWASMRegistration},
	{"DecodeWASMRegistrationRequest", DecodeWASMRegistrationRequest},
	{"DecodeWCEnvelope", DecodeWCEnvelope},
}

// hostileBodies returns byte frames built to drive a decoder's length fields to
// values that widen negative on a 32-bit int.
//
// The offsets are swept rather than targeted because the frame layouts differ
// per op and the point is to reach length fields wherever they sit. 0xFFFFFFFF
// widens to -1; 0x80000000 to MinInt32, which is the case that also breaks
// naive `n*elemSize` arithmetic. Small bodies catch the decoders that read a
// length before checking there is a body at all.
func hostileBodies() [][]byte {
	var out [][]byte
	patterns := []uint32{0xFFFFFFFF, 0x80000000, 0x7FFFFFFF, 0xFFFFFFFE}
	for _, size := range []int{0, 1, 4, 8, 16, 32, 64} {
		for _, p := range patterns {
			for off := 0; off+4 <= size; off += 4 {
				b := make([]byte, size)
				for i := range b {
					b[i] = 0xFF
				}
				binary.BigEndian.PutUint32(b[off:], p)
				out = append(out, b)
			}
			// An all-pattern frame: every 4-byte window is hostile at once.
			b := make([]byte, size)
			for off := 0; off+4 <= size; off += 4 {
				binary.BigEndian.PutUint32(b[off:], p)
			}
			out = append(out, b)
		}
		out = append(out, make([]byte, size)) // all zeroes
	}
	// Zero-based frames with ONE hostile window swept byte-by-byte. These reach a
	// length field that sits BEHIND a small/zero preamble (e.g. a second or third
	// length field). The all-0xFF frames above cannot: a 0xFF preamble makes the
	// FIRST length huge and the decoder bails before the later field. Stepping by
	// 1 (not 4) lands on unaligned fields too.
	for _, size := range []int{12, 16, 24, 32, 48} {
		for _, p := range patterns {
			for off := 0; off+4 <= size; off++ {
				b := make([]byte, size)
				binary.BigEndian.PutUint32(b[off:], p)
				out = append(out, b)
			}
		}
	}
	return out
}

func TestNoDecoderPanicsOnHostileBytes(t *testing.T) {
	// A floor on the roster, because the failure mode of this sweep is silent:
	// deleting entries makes it pass faster, not fail. Raise it when decoders are
	// added; only lower it deliberately, when an op is genuinely removed.
	const minDecoders = 137
	if len(hostileDecoders) < minDecoders {
		t.Fatalf("only %d decoders in the sweep (floor %d) — entries were removed, "+
			"which narrows the coverage without failing anything", len(hostileDecoders), minDecoders)
	}

	bodies := hostileBodies()
	for _, d := range hostileDecoders {
		fn := reflect.ValueOf(d.fn)
		if fn.Kind() != reflect.Func || fn.Type().NumIn() != 1 {
			t.Fatalf("%s: not a single-argument decoder", d.name)
		}
		for i, body := range bodies {
			func() {
				defer func() {
					if r := recover(); r != nil {
						t.Errorf("%s panicked on hostile body #%d (len %d): %v\n"+
							"a decoder must REJECT untrusted bytes, never panic on them — "+
							"on 32-bit this is usually an int(uint32) length that widened negative "+
							"and slipped past a len(args) < off+n check",
							d.name, i, len(body), r)
					}
				}()
				arg := make([]byte, len(body))
				copy(arg, body)
				fn.Call([]reflect.Value{reflect.ValueOf(arg)})
			}()
		}
	}
}

// TestHostileBodiesAreActuallyHostile guards the guard: if hostileBodies ever
// stops producing frames whose length fields exceed MaxInt32, the sweep above
// would pass vacuously while testing nothing.
func TestHostileBodiesAreActuallyHostile(t *testing.T) {
	var overMaxInt32 int
	for _, b := range hostileBodies() {
		for off := 0; off+4 <= len(b); off += 4 {
			if binary.BigEndian.Uint32(b[off:]) > 1<<31-1 {
				overMaxInt32++
			}
		}
	}
	if overMaxInt32 == 0 {
		t.Fatal("no hostile body carries a length above MaxInt32 — the sweep is vacuous")
	}
	t.Logf("%d length fields above MaxInt32 across %d bodies", overMaxInt32, len(hostileBodies()))
}
