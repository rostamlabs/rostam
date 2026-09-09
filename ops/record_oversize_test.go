// SPDX-License-Identifier: Apache-2.0

package ops

import (
	"errors"
	"testing"

	"github.com/rostamlabs/rostam/sdk/vtypes"
	"github.com/rostamlabs/rostam/vector"
)

// overCapRecordBytes is one byte past the engine's ValueRecord cap
// (vector's unexported maxRecordValueBytes, 16 MiB — the snapshot/WAL codec's
// limit). Spelled out rather than imported because the constant is internal to
// the vector package; if it ever moves, this test fails loudly by ACCEPTING the
// insert, which is the direction that matters.
const overCapRecordBytes = 16<<20 + 1

// TestVectorInsertOversizeRecordRejectedOnBothWirePaths is the regression for the
// gap the scoped re-review found: handleVectorInsert has TWO downstream paths,
// and only one of them was bounded.
//
// A non-zero `version` in the decoded wire args routes the insert to
// Collection.RestoreInsert/RestoreInsertAt — the version-preserving reinsert the
// reshard backfill uses — instead of InsertCASKeyTTL. That branch carries the
// CALLER's metadata, so it is an ingest path reachable from any client that sets
// a version, not the replay-only primitive its doc comment reads like. It used to
// accept a 16 MiB+1 record that the ordinary insert refused, and because
// Collection.RestoreInsert applies to the index BEFORE it appends to the WAL, the
// point went live and the append then failed — after which the collection could
// no longer be snapshotted.
//
// Both branches must answer the same way, so the test drives both through the
// real op handler.
func TestVectorInsertOversizeRecordRejectedOnBothWirePaths(t *testing.T) {
	tx := newVecTx(t)
	cfg := vector.Config{Dim: 2, M: 4, EfConstruction: 10, EfSearch: 10, Seed: 1, Metric: vector.L2}
	if _, err := handleVectorCreateCollection(tx, EncodeCreateCollectionArgs("docs", cfg)); err != nil {
		t.Fatalf("create collection: %v", err)
	}
	oversize := vtypes.Metadata{"session": vtypes.NewRecord(make([]byte, overCapRecordBytes))}
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
				"docs", tc.id, vec, 0, oversize, vtypes.SparseVector{}, tc.version))
			if err == nil {
				t.Fatal("insert with a 16 MiB+1 record: got nil error, want ErrRecordTooLarge")
			}
			if !errors.Is(err, vector.ErrRecordTooLarge) {
				t.Fatalf("err = %v, want vector.ErrRecordTooLarge", err)
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

	// The consequence the bound exists for: an ordinary insert alongside the
	// rejected ones still works, and the collection is still writable.
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
