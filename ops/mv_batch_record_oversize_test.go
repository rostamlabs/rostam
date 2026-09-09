// SPDX-License-Identifier: Apache-2.0

package ops

import (
	"errors"
	"io"
	"testing"

	"github.com/rostamlabs/rostam/sdk/vtypes"
	"github.com/rostamlabs/rostam/vector"
)

// mvBatchTx creates a fresh TxContext with one MV collection named "mv".
func mvBatchTx(t *testing.T) *TxContext {
	t.Helper()
	tx := newVecTx(t)
	cfg := vector.MultiVectorConfig{Dim: 2, M: 8, EfConstruction: 50, EfSearch: 32, Seed: 1}
	if _, err := handleMVCreate(tx, EncodeMVCreateArgs("mv", cfg)); err != nil {
		t.Fatalf("create MV collection: %v", err)
	}
	return tx
}

func mvRec(id uint64, meta vtypes.Metadata) vtypes.MultiScanRecord {
	return vtypes.MultiScanRecord{ID: id, Tokens: [][]float32{{1, 0}, {0, 1}}, Metadata: meta, Version: 1}
}

// TestMVAddBatchOversizeRecordRejectedOnBothPaths is the regression for the gap
// the deep review found: vector_mv_add_batch has TWO downstream paths and only
// one of them was bounded.
//
// handleMVAddBatch calls MultiBulkBuild as the FAST PATH whenever the target
// index is empty, falling through to the gated MultiRestoreAddSparse only once
// documents exist. MultiBulkBuild validated tokens, dimension and sparse but not
// records, so the same wire op was gated or ungated depending purely on the
// target's emptiness — and the ungated case, a fresh partition, is exactly the
// one an offline MV resplit drives.
//
// The damage is durability, not memory: MultiBulkBuild does not WAL-log
// (durability rides the replicated batch op), so an oversize record simply goes
// live and then every snapshot of the collection fails for as long as the
// document stays live. Both branches must answer the same way, so the test
// drives both through the real op handler.
func TestMVAddBatchOversizeRecordRejectedOnBothPaths(t *testing.T) {
	oversize := vtypes.Metadata{"session": vtypes.NewRecord(make([]byte, overCapRecordBytes))}

	t.Run("empty target (MultiBulkBuild fast path)", func(t *testing.T) {
		tx := mvBatchTx(t)
		_, err := handleMVAddBatch(tx, EncodeMVAddBatchArgs("mv", []vtypes.MultiScanRecord{
			mvRec(1, vtypes.Metadata{"rc": vtypes.NewInt(1)}),
			mvRec(2, oversize),
		}))
		if err == nil {
			t.Fatal("batch with a 16 MiB+1 record into an EMPTY index: got nil error, want ErrRecordTooLarge")
		}
		if !errors.Is(err, vector.ErrRecordTooLarge) {
			t.Fatalf("err = %v, want vector.ErrRecordTooLarge", err)
		}

		// The WHOLE batch is refused, including the good record that preceded the
		// bad one: validation runs before the index lock is taken, so nothing was
		// half-built.
		for _, id := range []uint64{1, 2} {
			eb, eerr := handleMVExists(tx, EncodeMVExistsArgs("mv", id))
			if eerr != nil {
				t.Fatalf("mv exists %d: %v", id, eerr)
			}
			if ex, _ := DecodeExistsResult(eb); ex {
				t.Fatalf("rejected batch still created document %d", id)
			}
		}

		// The consequence the bound exists for: the collection is still
		// snapshottable. An oversize record stored here would make writeValue
		// fail for every snapshot while the document stayed live.
		if err := tx.Vectors().SnapshotAll(io.Discard); err != nil {
			t.Fatalf("snapshot after the rejected batch: %v, want nil", err)
		}

		// And the index is still EMPTY, so a legitimate batch still takes the
		// fast path it was built for.
		if _, err := handleMVAddBatch(tx, EncodeMVAddBatchArgs("mv", []vtypes.MultiScanRecord{
			mvRec(3, vtypes.Metadata{"rc": vtypes.NewInt(2)}),
		})); err != nil {
			t.Fatalf("legitimate batch after the rejected one: %v, want nil", err)
		}
		eb, eerr := handleMVExists(tx, EncodeMVExistsArgs("mv", 3))
		if eerr != nil {
			t.Fatalf("mv exists 3: %v", eerr)
		}
		if ex, _ := DecodeExistsResult(eb); !ex {
			t.Fatal("the legitimate batch did not land")
		}
		if err := tx.Vectors().SnapshotAll(io.Discard); err != nil {
			t.Fatalf("snapshot after the legitimate batch: %v, want nil", err)
		}
	})

	t.Run("non-empty target (MultiRestoreAddSparse path)", func(t *testing.T) {
		tx := mvBatchTx(t)
		// Seed one document so the fast path declines and the batch falls
		// through to the per-record path, which was already gated. Both
		// branches must agree.
		if _, err := handleMVAddBatch(tx, EncodeMVAddBatchArgs("mv", []vtypes.MultiScanRecord{
			mvRec(1, vtypes.Metadata{"rc": vtypes.NewInt(1)}),
		})); err != nil {
			t.Fatalf("seed batch: %v", err)
		}
		_, err := handleMVAddBatch(tx, EncodeMVAddBatchArgs("mv", []vtypes.MultiScanRecord{mvRec(2, oversize)}))
		if err == nil {
			t.Fatal("batch with a 16 MiB+1 record into a NON-EMPTY index: got nil error, want ErrRecordTooLarge")
		}
		if !errors.Is(err, vector.ErrRecordTooLarge) {
			t.Fatalf("err = %v, want vector.ErrRecordTooLarge", err)
		}
		eb, eerr := handleMVExists(tx, EncodeMVExistsArgs("mv", 2))
		if eerr != nil {
			t.Fatalf("mv exists 2: %v", eerr)
		}
		if ex, _ := DecodeExistsResult(eb); ex {
			t.Fatal("rejected batch still created document 2")
		}
		if err := tx.Vectors().SnapshotAll(io.Discard); err != nil {
			t.Fatalf("snapshot after the rejected batch: %v, want nil", err)
		}
	})
}
