// SPDX-License-Identifier: Apache-2.0

package ops

import (
	"bytes"
	"encoding/binary"
	"errors"
	"reflect"
	"testing"

	"github.com/rostamlabs/rostam/sdk/vtypes"
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
	{"DecodeCAEXArgs", DecodeCAEXArgs},
	{"DecodeGetDelResult", DecodeGetDelResult},
	{"DecodeTTLResult", DecodeTTLResult},
	{"DecodeIncrExArgs", DecodeIncrExArgs},
	{"DecodeMGetArgs", DecodeMGetArgs},
	{"DecodeMGetResult", DecodeMGetResult},
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
	{"DecodeOperateArgs", DecodeOperateArgs},
	{"DecodeOperateResult", DecodeOperateResult},
	{"DecodePayloadResult", DecodePayloadResult},
	{"DecodePutArgs", DecodePutArgs},
	{"DecodePutBatchArgs", DecodePutBatchArgs},
	{"DecodePutBatchResult", DecodePutBatchResult},
	{"DecodeQueryArgs", DecodeQueryArgs},
	{"DecodeQueryResult", DecodeQueryResult},
	{"DecodeQueryResultDegraded", DecodeQueryResultDegraded},
	{"DecodeQueryResultGroupedFanOut", DecodeQueryResultGroupedFanOut},
	{"DecodeQuerySpecArgs", DecodeQuerySpecArgs},
	{"DecodeQueryTreeLanes", DecodeQueryTreeLanes},
	{"DecodeRecord", DecodeRecord},
	{"DecodeReshardAbortArgs", DecodeReshardAbortArgs},
	{"DecodeReshardArgs", DecodeReshardArgs},
	{"DecodeResplitArgs", DecodeResplitArgs},
	{"DecodeResplitCleanupArgs", DecodeResplitCleanupArgs},
	{"DecodeResplitCleanupResult", DecodeResplitCleanupResult},
	{"DecodeScanVectorsArgs", DecodeScanVectorsArgs},
	{"DecodeScanVectorsResult", DecodeScanVectorsResult},
	{"DecodeSchema", DecodeSchema},
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
	{"DecodeVectorOperateArgs", DecodeVectorOperateArgs},
	{"DecodeVectorOperateResult", DecodeVectorOperateResult},
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
	const minDecoders = 150
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

// --- Gated-branch coverage (Finding B) ---------------------------------------
//
// hostileBodies above emits zero-padded single-window frames. Those cannot enter
// decoder branches gated behind nonzero presence/kind/flag bytes: the order
// block's string-resume tail needs order-present + string-kind + resume-present
// markers set, and its multi-key tail needs the multi-key flag and a nonzero
// numTail. A length field living INSIDE such a tail is therefore invisible to
// the zero-padded sweep — which is exactly how the additive off+sl / off+kl2+1
// checks in readScrollOrderBlock slipped past it.
//
// The seeds below are VALID order frames built with the package's own encoders
// with the gating bytes set, so the shared readScrollOrderBlock reaches each
// tail. hostileSeedMutants then drives every embedded length field
// negative/overflowing while leaving the gating bytes intact, so the branch that
// reads it stays reachable. Under GOARCH=386 these mutants panic on the
// pre-fix additive checks and are rejected (ErrVectorArgsTruncated) on the fix.

const (
	resumeStrMark = "__RESUME_STR_MARK__"
	tailKeyMark   = "__TAIL_KEY_MARK__"
	cursorStrMark = "__CURSOR_STR_MARK__"
)

type scrollOrderSeed struct {
	name  string
	frame []byte
	fn    any
}

// scrollOrderSeeds returns valid order frames whose order block reaches each
// control-byte-gated length tail (string-resume, multi-key key, multi-key resume
// tuple string), paired with the family decoder that must survive every hostile
// mutant of them. All three families share readScrollOrderBlock, so each is
// exercised through its own base layout.
func scrollOrderSeeds() []scrollOrderSeed {
	filter := vtypes.Filter{}
	strOrder := &ScrollOrder{Key: "k", Kind: vtypes.OrderString, HasResumeStr: true, ResumeStr: resumeStrMark}
	mkOrder := &ScrollOrder{Key: "k", Tail: []ScrollOrderKey{{Key: tailKeyMark}}}
	tupOrder := &ScrollOrder{
		Key:           "k",
		Tail:          []ScrollOrderKey{{Key: "t", Kind: vtypes.OrderString}},
		HasResumeKeys: true,
		ResumeKeys: []ScrollOrderVal{
			{Str: "p", Kind: vtypes.OrderString},
			{Str: cursorStrMark, Kind: vtypes.OrderString},
		},
	}
	var out []scrollOrderSeed
	for _, o := range []struct {
		tag   string
		order *ScrollOrder
	}{{"strResume", strOrder}, {"multiKey", mkOrder}, {"tupleStr", tupOrder}} {
		out = append(out,
			scrollOrderSeed{"Dense/" + o.tag, EncodeScrollArgsOrder("c", filter, 10, 0, 0, 0, false, o.order), DecodeScrollArgsOrder},
			scrollOrderSeed{"Named/" + o.tag, EncodeNamedScrollArgsOrder("c", filter, 10, 0, false, 0, 0, o.order), DecodeNamedScrollArgsOrder},
			scrollOrderSeed{"MV/" + o.tag, EncodeMVScrollArgsOrder("c", filter, 10, 0, 0, 0, false, o.order), DecodeMVScrollArgsOrder},
		)
	}
	return out
}

// hostileSeedMutants overwrites every 4-byte window of seed (stepping by 1 to
// catch unaligned fields too) with each hostile pattern, leaving the rest of the
// frame — crucially the gating control bytes — untouched. Every embedded length
// field is therefore hit while the branch that reads it stays reachable.
func hostileSeedMutants(seed []byte) [][]byte {
	patterns := []uint32{0xFFFFFFFF, 0x80000000, 0x7FFFFFFF, 0xFFFFFFFE}
	var out [][]byte
	for _, p := range patterns {
		for off := 0; off+4 <= len(seed); off++ {
			b := make([]byte, len(seed))
			copy(b, seed)
			binary.BigEndian.PutUint32(b[off:], p)
			out = append(out, b)
		}
	}
	return out
}

func TestNoScrollOrderDecoderPanicsOnGatedHostileBytes(t *testing.T) {
	seeds := scrollOrderSeeds()
	if len(seeds) == 0 {
		t.Fatal("no scroll-order seeds built")
	}
	for _, s := range seeds {
		fn := reflect.ValueOf(s.fn)
		if fn.Kind() != reflect.Func || fn.Type().NumIn() != 1 {
			t.Fatalf("%s: not a single-argument decoder", s.name)
		}
		mutants := hostileSeedMutants(s.frame)
		for i, body := range mutants {
			func() {
				defer func() {
					if r := recover(); r != nil {
						t.Errorf("%s panicked on gated hostile mutant #%d (len %d): %v\n"+
							"a control-byte-gated order tail (string-resume / multi-key) length "+
							"widened negative on 32-bit and slipped past an additive off+n check",
							s.name, i, len(body), r)
					}
				}()
				arg := make([]byte, len(body))
				copy(arg, body)
				fn.Call([]reflect.Value{reflect.ValueOf(arg)})
			}()
		}
	}
}

// TestScrollOrderSeedMutantsAreHostile guards the gated sweep the way
// TestHostileBodiesAreActuallyHostile guards the main one: if the mutant
// generator ever stops emitting length fields above MaxInt32 it would pass
// vacuously.
func TestScrollOrderSeedMutantsAreHostile(t *testing.T) {
	var over int
	for _, s := range scrollOrderSeeds() {
		for _, b := range hostileSeedMutants(s.frame) {
			for off := 0; off+4 <= len(b); off += 4 {
				if binary.BigEndian.Uint32(b[off:]) > 1<<31-1 {
					over++
				}
			}
		}
	}
	if over == 0 {
		t.Fatal("no scroll-order seed mutant carries a length field above MaxInt32 — the gated sweep is vacuous")
	}
	t.Logf("%d length fields above MaxInt32 across scroll-order seed mutants", over)
}

// TestScrollOrderStringResumeHostileLenRejected targets the string-resume tail's
// strLen check (readScrollOrderBlock ~sl). A valid OrderString frame with the
// resume string present has its strLen driven to 0x7fffffff — the value that
// stays positive past the sl<0 guard and overflows off+sl on 32-bit. The decode
// must REJECT it, never panic, on 386 and native alike.
func TestScrollOrderStringResumeHostileLenRejected(t *testing.T) {
	order := &ScrollOrder{Key: "k", Kind: vtypes.OrderString, HasResumeStr: true, ResumeStr: resumeStrMark}
	frame := EncodeScrollArgsOrder("c", vtypes.Filter{}, 10, 0, 0, 0, false, order)
	mi := bytes.Index(frame, []byte(resumeStrMark))
	if mi < 4 {
		t.Fatalf("resume-str marker not found at a decodable offset (%d)", mi)
	}
	binary.BigEndian.PutUint32(frame[mi-4:], 0x7fffffff) // strLen field precedes the marker bytes
	_, _, _, _, _, _, _, _, err := DecodeScrollArgsOrder(frame)
	if !errors.Is(err, ErrVectorArgsTruncated) {
		t.Fatalf("string-resume sl=0x7fffffff: got err=%v, want ErrVectorArgsTruncated", err)
	}
}

// TestScrollOrderMultiKeyHostileLenRejected targets the multi-key tail's per-key
// keyLen check (readScrollOrderBlock ~kl2, the off+kl2+1 additive form). A valid
// multi-key frame with one tail key has that key's keyLen driven to 0x7fffffff.
// The decode must REJECT it, never panic, on 386 and native alike.
func TestScrollOrderMultiKeyHostileLenRejected(t *testing.T) {
	order := &ScrollOrder{Key: "k", Tail: []ScrollOrderKey{{Key: tailKeyMark}}}
	frame := EncodeScrollArgsOrder("c", vtypes.Filter{}, 10, 0, 0, 0, false, order)
	mi := bytes.Index(frame, []byte(tailKeyMark))
	if mi < 4 {
		t.Fatalf("tail-key marker not found at a decodable offset (%d)", mi)
	}
	binary.BigEndian.PutUint32(frame[mi-4:], 0x7fffffff) // keyLen field precedes the marker bytes
	_, _, _, _, _, _, _, _, err := DecodeScrollArgsOrder(frame)
	if !errors.Is(err, ErrVectorArgsTruncated) {
		t.Fatalf("multi-key kl2=0x7fffffff: got err=%v, want ErrVectorArgsTruncated", err)
	}
}

// TestResultDecoderDimOverflowRejected covers the inner per-element `dim` length
// in three result decoders (DecodeMatrix / DecodeScanVectorsResult /
// DecodeNamedGetResultAt), whose `len < off+4*dim` checks were multiplicative,
// not the additive `off+n` form the main sweep and the scroll-order seeds target.
// On 386 a hostile dim = 0x40000000 makes 4*dim widen to 0, slip the old check,
// and panic make([]float32, dim). These frames set a VALID outer count so decode
// reaches the inner dim — the zero-padded sweep rejects at the outer count and
// never gets here. Post-fix (CountFitsIn divide-form) each must reject, not panic.
func TestResultDecoderDimOverflowRejected(t *testing.T) {
	// DecodeMatrix: [rows=1][dim=0x40000000]
	if _, _, err := DecodeMatrix([]byte{0, 0, 0, 1, 0x40, 0, 0, 0}); err == nil {
		t.Error("DecodeMatrix: hostile dim accepted, want error (must not panic on 386)")
	}
	// DecodeScanVectorsResult: [count=1][id:8][dim=0x40000000][pad:12]. len 28 so
	// the OLD `< off+4*dim+8+4` (== `< 28` when 4*dim overflows to 0) passes and
	// reaches make(); the fix rejects via CountFitsIn.
	scan := make([]byte, 28)
	scan[3] = 1     // count = 1 (body[0:4] big-endian)
	scan[12] = 0x40 // dim = 0x40000000 (body[12:16] big-endian, MSB)
	if _, err := DecodeScanVectorsResult(scan); err == nil {
		t.Error("DecodeScanVectorsResult: hostile dim accepted, want error (must not panic on 386)")
	}
	// DecodeNamedGetResultAt: [found=1][numSpaces=1][nameLen=0][dim=0x40000000]
	named := []byte{1, 0, 0, 0, 1, 0, 0, 0x40, 0, 0, 0}
	if _, _, _, _, _, _, err := DecodeNamedGetResultAt(named, 0); err == nil {
		t.Error("DecodeNamedGetResultAt: hostile dim accepted, want error (must not panic on 386)")
	}
}
