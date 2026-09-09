// SPDX-License-Identifier: Apache-2.0

package ops

import (
	"errors"
	"testing"

	"github.com/rostamlabs/rostam/sdk/vtypes"
	"github.com/rostamlabs/rostam/vector"
)

// malformedRecordBytes is a schema mode byte followed by a torn schema blob: no
// operate engine can open it, and IndexEntries cannot enumerate it while Resolve
// still answers from a record shaped like it — the asymmetry that poisons a
// payload key for the whole collection.
var malformedRecordBytes = []byte{0x01, 0xFF}

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
