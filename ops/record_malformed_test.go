// SPDX-License-Identifier: Apache-2.0

package ops

import (
	"errors"
	"testing"

	"github.com/rostamlabs/rostam/sdk/vtypes"
	"github.com/rostamlabs/rostam/sdk/wire"
	"github.com/rostamlabs/rostam/vector"
)

// malformedRecordBytes is a schema mode byte followed by a torn schema blob: no
// operate engine can open it, and IndexEntries cannot enumerate it while Resolve
// still answers from a record shaped like it — the asymmetry that poisons a
// payload key for the whole collection.
var malformedRecordBytes = []byte{0x01, 0xFF}

// sessionRecordForOps is a small well-formed schema-mode record, so the tests
// below can show the gate accepting as well as rejecting.
func sessionRecordForOps(t *testing.T) []byte {
	t.Helper()
	sch := &wire.Schema{Version: 1, StoreNames: true, Fields: []wire.FieldDef{
		{Name: "rc", Type: wire.OperateTypeU8},
		{Name: "tag", Type: wire.OperateTypeBytes},
	}}
	if err := sch.Validate(); err != nil {
		t.Fatalf("schema invalid: %v", err)
	}
	enc := (&wire.Record{Mode: wire.OperateModeSchema, Schema: sch, Fields: []wire.Field{
		{Cell: wire.Cell{Type: wire.OperateTypeU8, U: 7}},
		{Cell: wire.Cell{Type: wire.OperateTypeBytes, B: []byte("de")}},
	}}).Encode()
	if enc == nil {
		t.Fatal("session record failed to encode")
	}
	return enc
}

// TestVectorInsertMalformedRecordRejectedOnBothWirePaths is the shape sibling of
// TestVectorInsertOversizeRecordRejectedOnBothWirePaths, and it exists for the
// same reason: handleVectorInsert has TWO downstream paths and both carry the
// CALLER's metadata.
//
// A non-zero `version` in the decoded wire args routes the insert to
// Collection.RestoreInsert/RestoreInsertAt, whose doc comment reads like a
// replay-only primitive but which any client that sets a version reaches. The
// phase-2 pre-flight scan called this out: the engine-level restore BODIES stay
// size-only so WAL replay can never refuse an acked write, while these two
// wire-reachable entries validate shape like every other ingest entry. Both
// branches must answer the same way, so the test drives both through the real op
// handler.
func TestVectorInsertMalformedRecordRejectedOnBothWirePaths(t *testing.T) {
	tx := newVecTx(t)
	cfg := vector.Config{Dim: 2, M: 4, EfConstruction: 10, EfSearch: 10, Seed: 1, Metric: vector.L2}
	if _, err := handleVectorCreateCollection(tx, EncodeCreateCollectionArgs("docs", cfg)); err != nil {
		t.Fatalf("create collection: %v", err)
	}
	malformed := vtypes.Metadata{"session": vtypes.NewRecord(malformedRecordBytes)}
	vec := []float32{1, 0}

	for _, tc := range []struct {
		name    string
		id      uint64
		version uint64
	}{
		{"ordinary insert (version 0)", 1, 0},
		{"version-preserving reinsert (version != 0)", 2, 7},
	} {
		t.Run(tc.name, func(t *testing.T) {
			_, err := handleVectorInsert(tx, EncodeVectorInsertArgsVersioned(
				"docs", tc.id, vec, 0, malformed, vtypes.SparseVector{}, tc.version))
			if err == nil {
				t.Fatal("insert with a malformed record: got nil error, want ErrRecordMalformed")
			}
			if !errors.Is(err, vector.ErrRecordMalformed) {
				t.Fatalf("err = %v, want vector.ErrRecordMalformed", err)
			}
			eb, eerr := handleVectorExists(tx, EncodeExistsArgs("docs", tc.id))
			if eerr != nil {
				t.Fatalf("exists: %v", eerr)
			}
			if ex, _ := DecodeExistsResult(eb); ex {
				t.Fatalf("rejected insert still created point %d", tc.id)
			}
		})
	}

	// The gate is not over-broad: the same version-preserving reinsert carrying
	// a well-formed payload still lands.
	if _, err := handleVectorInsert(tx, EncodeVectorInsertArgsVersioned(
		"docs", 3, vec, 0, vtypes.Metadata{"rc": vtypes.NewInt(1)}, vtypes.SparseVector{}, 7)); err != nil {
		t.Fatalf("version-preserving reinsert with an ordinary payload: %v, want nil", err)
	}
	eb, err := handleVectorExists(tx, EncodeExistsArgs("docs", 3))
	if err != nil {
		t.Fatalf("exists: %v", err)
	}
	if ex, _ := DecodeExistsResult(eb); !ex {
		t.Fatal("the legitimate version-preserving reinsert did not land")
	}
}

// TestMVAddBatchGatesRecordValuesOnBothPaths is the handler-level regression for
// the fix-round-1 finding: vector_mv_add_batch has TWO downstream paths and only
// one of them was gated.
//
// handleMVAddBatch calls MultiBulkBuild first — a concurrent whole-batch build
// that only runs when the target index is EMPTY — and falls through to the
// per-record MultiRestoreAddSparse when documents already exist. MultiBulkBuild
// wrote the caller's metadata into docMeta and reindexed it with no record gate
// at all, so the SAME op accepted or refused a malformed record depending on
// whether the target happened to be empty. The empty case is not the rare one: an
// offline MV resplit copies into fresh partitions, which is precisely it.
//
// Both paths, both bounds, are driven through the real handler.
func TestMVAddBatchGatesRecordValuesOnBothPaths(t *testing.T) {
	good := vtypes.Metadata{"session": vtypes.NewRecord(sessionRecordForOps(t))}
	tokens := [][]float32{{1, 0}}

	// BOTH ROWS REJECT INSIDE MultiBulkBuild, AND THAT IS THE POINT. Its
	// whole-batch validation runs BEFORE it looks at whether the target is empty
	// (vector/multivector.go: the record gate is in the loop above m.mu.Lock,
	// the emptiness check is after it), so through this handler a bad record
	// never reaches the per-record MultiRestoreAddSparse below. The seeded row
	// therefore pins that the answer does not depend on the target's state — the
	// exact asymmetry the fix closed — rather than exercising a second gate.
	// MultiRestoreAddSparse's own gate is driven directly by
	// TestMVAddVersionedGatesRecordValues, through the other wire op that
	// reaches it.
	for _, path := range []struct {
		name string
		seed bool // seed a document, so the target is non-empty
	}{
		{"empty target", false},
		{"non-empty target", true},
	} {
		for _, bad := range []struct {
			name string
			meta vtypes.Metadata
			want error
		}{
			{"malformed", vtypes.Metadata{"session": vtypes.NewRecord(malformedRecordBytes)}, vector.ErrRecordMalformed},
			{"oversize", vtypes.Metadata{"session": vtypes.NewRecord(make([]byte, overCapRecordBytes))}, vector.ErrRecordTooLarge},
		} {
			t.Run(path.name+"/"+bad.name, func(t *testing.T) {
				tx := newVecTx(t)
				cfg := vector.MultiVectorConfig{Dim: 2, M: 4, EfConstruction: 10, EfSearch: 10, Seed: 1}
				if _, err := handleMVCreate(tx, EncodeMVCreateArgs("mv", cfg)); err != nil {
					t.Fatalf("create MV: %v", err)
				}
				if path.seed {
					if _, err := handleMVAdd(tx, EncodeMVAddArgs("mv", 100, tokens, good)); err != nil {
						t.Fatalf("seed add: %v", err)
					}
				}
				// The bad record comes FIRST and a good one follows it, so a
				// gate that only looked at the first record and a gate that
				// walked the whole batch are told apart by the assertion below
				// that NEITHER document was stored.
				recs := []vtypes.MultiScanRecord{
					{ID: 1, Tokens: tokens, Metadata: bad.meta},
					{ID: 2, Tokens: tokens, Metadata: good},
				}
				_, err := handleMVAddBatch(tx, EncodeMVAddBatchArgs("mv", recs))
				if err == nil {
					t.Fatalf("batch carrying a %s record: got nil error, want %v", bad.name, bad.want)
				}
				if !errors.Is(err, bad.want) {
					t.Fatalf("err = %v, want %v", err, bad.want)
				}
				for _, id := range []uint64{1, 2} {
					eb, eerr := handleMVExists(tx, EncodeMVExistsArgs("mv", id))
					if eerr != nil {
						t.Fatalf("mv exists(%d): %v", id, eerr)
					}
					if ex, _ := DecodeExistsResult(eb); ex {
						t.Errorf("refused batch still stored doc %d", id)
					}
				}
				// The gate is not over-broad: an all-good batch still lands on
				// whichever path this case exercises.
				if _, err := handleMVAddBatch(tx, EncodeMVAddBatchArgs("mv", []vtypes.MultiScanRecord{
					{ID: 3, Tokens: tokens, Metadata: good},
				})); err != nil {
					t.Fatalf("all-good batch: %v, want nil", err)
				}
				eb, eerr := handleMVExists(tx, EncodeMVExistsArgs("mv", 3))
				if eerr != nil {
					t.Fatalf("mv exists(3): %v", eerr)
				}
				if ex, _ := DecodeExistsResult(eb); !ex {
					t.Fatal("the all-good batch did not land")
				}
			})
		}
	}
}

// TestMVAddVersionedGatesRecordValues drives MultiRestoreAddSparse's OWN record
// gate through the wire op that actually reaches it.
//
// vector_mv_add_batch cannot: handleMVAddBatch calls MultiBulkBuild first, and
// that validates the whole batch before it checks whether the target is empty,
// so a malformed record is refused there and the per-record restore-add below is
// never entered. vector_mv_add_versioned — the offline MV-resplit backfill
// primitive — calls MultiRestoreAddSparse directly, which is where the gate that
// protects the incremental path has to be proven.
//
// A refused record must leave the document absent: the gate runs before any
// state change, not after a partial one.
func TestMVAddVersionedGatesRecordValues(t *testing.T) {
	tokens := [][]float32{{1, 0}}
	good := vtypes.Metadata{"session": vtypes.NewRecord(sessionRecordForOps(t))}

	for _, bad := range []struct {
		name string
		meta vtypes.Metadata
		want error
	}{
		{"malformed", vtypes.Metadata{"session": vtypes.NewRecord(malformedRecordBytes)}, vector.ErrRecordMalformed},
		{"oversize", vtypes.Metadata{"session": vtypes.NewRecord(make([]byte, overCapRecordBytes))}, vector.ErrRecordTooLarge},
	} {
		t.Run(bad.name, func(t *testing.T) {
			tx := newVecTx(t)
			cfg := vector.MultiVectorConfig{Dim: 2, M: 4, EfConstruction: 10, EfSearch: 10, Seed: 1}
			if _, err := handleMVCreate(tx, EncodeMVCreateArgs("mv", cfg)); err != nil {
				t.Fatalf("create MV: %v", err)
			}
			// A seeded document, so this is the INCREMENTAL path and not a
			// bulk build.
			if _, err := handleMVAdd(tx, EncodeMVAddArgs("mv", 100, tokens, good)); err != nil {
				t.Fatalf("seed add: %v", err)
			}

			_, err := handleMVAddVersioned(tx, EncodeMVAddArgsVersioned("mv", 1, tokens, bad.meta, 7))
			if err == nil {
				t.Fatalf("versioned add carrying a %s record: got nil error, want %v", bad.name, bad.want)
			}
			if !errors.Is(err, bad.want) {
				t.Fatalf("err = %v, want %v", err, bad.want)
			}
			eb, eerr := handleMVExists(tx, EncodeMVExistsArgs("mv", 1))
			if eerr != nil {
				t.Fatalf("mv exists(1): %v", eerr)
			}
			if ex, _ := DecodeExistsResult(eb); ex {
				t.Error("the refused versioned add still stored the document")
			}

			// The gate is not over-broad: a good record on the same path lands,
			// with the verbatim version the op exists to preserve.
			if _, err := handleMVAddVersioned(tx, EncodeMVAddArgsVersioned("mv", 2, tokens, good, 7)); err != nil {
				t.Fatalf("all-good versioned add: %v, want nil", err)
			}
			eb, eerr = handleMVExists(tx, EncodeMVExistsArgs("mv", 2))
			if eerr != nil {
				t.Fatalf("mv exists(2): %v", eerr)
			}
			if ex, _ := DecodeExistsResult(eb); !ex {
				t.Fatal("the all-good versioned add did not land")
			}
		})
	}
}
