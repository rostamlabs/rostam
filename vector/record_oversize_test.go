// SPDX-License-Identifier: Apache-2.0

package vector

import (
	"bytes"
	"errors"
	"os"
	"path/filepath"
	"testing"
)

// A ValueRecord above maxRecordValueBytes is the one payload value the
// snapshot/WAL codec cannot write (writeValue rejects it), and it used to be
// ACCEPTED at ingest. The consequences were not local: writeOptMeta discarded
// writeValue's error after the kind byte had already been emitted, so the WAL
// held a self-inconsistent meta block; on the next replay readOptMeta answered
// false, replay returned errStopReplay, and every ACKNOWLEDGED write logged
// after that record was discarded. Meanwhile every snapshot of the collection
// failed for as long as the point stayed live. A base64 body of 17 MiB fits the
// server's 32 MiB request cap, so the whole shape was remotely reachable.
//
// These tests pin the fix from both ends: rejected at every ingest entry
// (checkRecordValues), and — as defence in depth for any path that reaches the
// codec anyway — the WAL append fails BEFORE a byte of the half-built record
// reaches the log.

// oversizeRecord builds a payload whose record value is one byte past the cap.
// It is not a valid record — it never needs to be, because the size check runs
// before anything looks inside it.
func oversizeRecord() Metadata {
	return Metadata{"session": NewRecord(make([]byte, maxRecordValueBytes+1))}
}

// atCapRecord is the largest record the codec accepts: the boundary must be
// inclusive, or the check would move the limit rather than enforce it.
func atCapRecord() Metadata {
	return Metadata{"session": NewRecord(make([]byte, maxRecordValueBytes))}
}

func wantTooLarge(t *testing.T, what string, err error) {
	t.Helper()
	if err == nil {
		t.Fatalf("%s: got nil error, want ErrRecordTooLarge", what)
	}
	if !errors.Is(err, ErrRecordTooLarge) {
		t.Fatalf("%s: err = %v, want ErrRecordTooLarge", what, err)
	}
}

func oversizeDenseConfig() Config {
	return Config{Dim: 4, M: 8, EfConstruction: 50, EfSearch: 50, Seed: 1}
}

func oversizeIVFConfig() Config {
	return Config{Dim: 4, M: 8, EfConstruction: 50, EfSearch: 50, Seed: 1,
		IndexType: IndexIVF, IVFNlist: 4, IVFNprobe: 2}
}

// TestOversizeRecordRejectedDense covers the dense (HNSW) entry-point family:
// insert, insert-if-absent, upsert, set-payload, overwrite and the bulk
// stage/build path. Each must reject with ErrRecordTooLarge and leave the
// point exactly as it was.
func TestOversizeRecordRejectedDense(t *testing.T) {
	for _, tc := range []struct {
		name string
		cfg  Config
	}{{"hnsw", oversizeDenseConfig()}, {"ivf", oversizeIVFConfig()}} {
		t.Run(tc.name, func(t *testing.T) {
			c, err := NewCollection("default/c", tc.cfg)
			if err != nil {
				t.Fatalf("NewCollection: %v", err)
			}
			vec := []float32{1, 0, 0, 0}
			good := Metadata{"rc": NewInt(1)}
			if err := c.Insert(1, vec, 0, good, nil); err != nil {
				t.Fatalf("baseline Insert: %v", err)
			}

			wantTooLarge(t, "Insert", c.Insert(2, vec, 0, oversizeRecord(), nil))
			if _, _, _, _, _, ok := c.Get(2); ok {
				t.Fatal("rejected Insert still created point 2")
			}

			_, err = c.InsertIfAbsent(3, vec, 0, oversizeRecord(), nil)
			wantTooLarge(t, "InsertIfAbsent", err)
			if _, _, _, _, _, ok := c.Get(3); ok {
				t.Fatal("rejected InsertIfAbsent still created point 3")
			}

			wantTooLarge(t, "Upsert", c.Upsert(1, vec, "doc", 0, oversizeRecord(), nil))
			// Upsert is delete-then-insert: the point must survive a rejection.
			_, meta, _, _, _, ok := c.Get(1)
			if !ok {
				t.Fatal("rejected Upsert deleted the existing point 1")
			}
			if got, gok := meta["rc"]; !gok || !got.Equal(NewInt(1)) {
				t.Fatalf("point 1 payload changed after a rejected Upsert: %v", meta)
			}

			wantTooLarge(t, "SetPayload", c.SetPayload(1, oversizeRecord(), nil))
			wantTooLarge(t, "OverwritePayload", c.OverwritePayload(1, oversizeRecord(), nil))
			_, meta, _, _, _, _ = c.Get(1)
			if got, gok := meta["rc"]; !gok || !got.Equal(NewInt(1)) {
				t.Fatalf("point 1 payload changed after a rejected payload op: %v", meta)
			}
			if _, gok := meta["session"]; gok {
				t.Fatal("rejected payload op still stored the oversize record")
			}

			// Bulk: the stage refuses the batch outright, so nothing is queued.
			bulk, berr := NewCollection("default/bulk", tc.cfg)
			if berr != nil {
				t.Fatalf("NewCollection bulk: %v", berr)
			}
			wantTooLarge(t, "StageBulkPayloads", bulk.StageBulkPayloads(
				[]uint64{9}, [][]float32{vec}, []Metadata{oversizeRecord()}))
			wantTooLarge(t, "BuildConcurrentMeta", bulk.BuildConcurrentMeta(
				[]uint64{9}, [][]float32{vec}, []Metadata{oversizeRecord()}, 1))
			if n := bulk.Stats().Size; n != 0 {
				t.Fatalf("rejected bulk build left %d points", n)
			}
		})
	}
}

// TestOversizeRecordRejectedNamed covers the named-vector family.
func TestOversizeRecordRejectedNamed(t *testing.T) {
	nc, err := NewNamedCollection("default/named", namedTestConfig())
	if err != nil {
		t.Fatalf("NewNamedCollection: %v", err)
	}
	vecs := map[string][]float32{"title": {1, 0, 0, 0}, "image": {1, 0, 0}}
	good := Metadata{"rc": NewInt(1)}
	if err := nc.Insert(1, vecs, good, 0); err != nil {
		t.Fatalf("baseline Insert: %v", err)
	}

	wantTooLarge(t, "NamedInsert", nc.Insert(2, vecs, oversizeRecord(), 0))
	if _, _, _, _, ok := nc.Get(2); ok {
		t.Fatal("rejected named Insert still created point 2")
	}
	wantTooLarge(t, "NamedSetPayload", nc.SetPayload(1, oversizeRecord(), nil))
	wantTooLarge(t, "NamedOverwritePayload", nc.OverwritePayload(1, oversizeRecord(), nil))
	_, payload, _, _, ok := nc.Get(1)
	if !ok {
		t.Fatal("point 1 disappeared")
	}
	if got, gok := payload["rc"]; !gok || !got.Equal(NewInt(1)) {
		t.Fatalf("named payload changed after a rejected op: %v", payload)
	}
	if _, gok := payload["session"]; gok {
		t.Fatal("rejected named payload op still stored the oversize record")
	}
}

// TestOversizeRecordRejectedMultiVector covers the multi-vector family,
// including the if-absent path that delegates to AddIfAbsent.
func TestOversizeRecordRejectedMultiVector(t *testing.T) {
	m, err := NewMultiVectorIndex(MultiVectorConfig{Dim: 4, M: 16, EfConstruction: 200, EfSearch: 64, Seed: 1})
	if err != nil {
		t.Fatalf("NewMultiVectorIndex: %v", err)
	}
	tokens := [][]float32{{1, 0, 0, 0}}
	good := Metadata{"rc": NewInt(1)}
	if err := m.Add(1, tokens, good); err != nil {
		t.Fatalf("baseline Add: %v", err)
	}

	wantTooLarge(t, "MVAdd", m.Add(2, tokens, oversizeRecord()))
	if _, _, _, ok := m.Get(2); ok {
		t.Fatal("rejected MV Add still created doc 2")
	}
	_, err = m.AddIfAbsent(3, tokens, oversizeRecord())
	wantTooLarge(t, "MVAddIfAbsent", err)
	if _, _, _, ok := m.Get(3); ok {
		t.Fatal("rejected MV AddIfAbsent still created doc 3")
	}
	// The versioned if-absent path does NOT delegate to AddIfAbsent (version != 0).
	_, err = m.MultiAddIfAbsentVersion(4, tokens, oversizeRecord(), nil, 5)
	wantTooLarge(t, "MVMultiAddIfAbsentVersion", err)
	if _, _, _, ok := m.Get(4); ok {
		t.Fatal("rejected MV MultiAddIfAbsentVersion still created doc 4")
	}

	wantTooLarge(t, "MVSetPayload", m.SetPayload(1, oversizeRecord(), nil))
	wantTooLarge(t, "MVOverwritePayload", m.OverwritePayload(1, oversizeRecord(), nil))
	_, payload, _, ok := m.Get(1)
	if !ok {
		t.Fatal("doc 1 disappeared")
	}
	if got, gok := payload["rc"]; !gok || !got.Equal(NewInt(1)) {
		t.Fatalf("MV payload changed after a rejected op: %v", payload)
	}
	if _, gok := payload["session"]; gok {
		t.Fatal("rejected MV payload op still stored the oversize record")
	}
}

// TestRecordAtCapAccepted pins the boundary: exactly maxRecordValueBytes is
// still a legal value, so the check rejects only what the codec rejects.
func TestRecordAtCapAccepted(t *testing.T) {
	c, err := NewCollection("default/cap", oversizeDenseConfig())
	if err != nil {
		t.Fatalf("NewCollection: %v", err)
	}
	if err := c.Insert(1, []float32{1, 0, 0, 0}, 0, atCapRecord(), nil); err != nil {
		t.Fatalf("Insert at exactly the cap: %v, want nil", err)
	}
}

// TestWriteOptMetaOversizeFailsAppend is the durability half. writeOptMeta must
// report the encode failure, and every staged append must fail on it BEFORE the
// record reaches the log — a partially written meta block is what replay reads
// as a torn record, and it answers by discarding every acked write behind it.
func TestWriteOptMetaOversizeFailsAppend(t *testing.T) {
	t.Run("codec returns the error", func(t *testing.T) {
		var buf bytes.Buffer
		if err := writeOptMeta(&buf, oversizeRecord()); err == nil {
			t.Fatal("writeOptMeta on an oversize record: got nil error, want one")
		}
	})

	path := filepath.Join(t.TempDir(), "w.wal")
	w, err := openWAL(path, true)
	if err != nil {
		t.Fatalf("openWAL: %v", err)
	}
	defer func() { _ = w.close() }()

	vec := []float32{1, 0, 0, 0}
	// A good record first, so the test proves the FAILED append adds nothing
	// rather than that the file happens to be empty.
	if _, err := w.appendInsertStaged(1, vec, 0, Metadata{"rc": NewInt(1)}, nil, nil, 1); err != nil {
		t.Fatalf("baseline appendInsertStaged: %v", err)
	}
	size := func() int64 {
		fi, serr := os.Stat(path)
		if serr != nil {
			t.Fatalf("stat: %v", serr)
		}
		return fi.Size()
	}
	before := size()
	if before == 0 {
		t.Fatal("baseline append wrote nothing")
	}

	for _, tc := range []struct {
		name string
		run  func() (uint64, error)
	}{
		{"insert", func() (uint64, error) {
			return w.appendInsertStaged(2, vec, 0, oversizeRecord(), nil, nil, 1)
		}},
		{"setPayload", func() (uint64, error) {
			return w.appendSetPayloadStaged(1, oversizeRecord(), nil, 2)
		}},
		{"namedInsert", func() (uint64, error) {
			return w.appendNamedInsertStaged(3, map[string][]float32{"t": vec}, nil, 0, oversizeRecord(), nil, 1)
		}},
		{"namedSetPayload", func() (uint64, error) {
			return w.appendNamedSetPayloadStaged(1, oversizeRecord(), nil, 2)
		}},
		{"mvAdd", func() (uint64, error) {
			return w.appendMVAddStaged(4, [][]float32{vec}, oversizeRecord(), nil, 1, nil)
		}},
		{"mvSetPayload", func() (uint64, error) {
			return w.appendMVSetPayloadStaged(1, oversizeRecord(), nil, 2)
		}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			seq, aerr := tc.run()
			if aerr == nil {
				t.Fatal("staged append with an oversize record: got nil error, want one")
			}
			if seq != 0 {
				t.Fatalf("failed append staged seq %d, want 0 (nothing may be staged)", seq)
			}
			if got := size(); got != before {
				t.Fatalf("WAL grew from %d to %d bytes on a failed append", before, got)
			}
		})
	}
}

// TestOversizeRecordLeavesSnapshotWritable is the reviewer's scenario end to
// end: the oversize insert is refused, and because it never landed, an ordinary
// insert alongside it still snapshots. Before the fix the refused point was
// live and every snapshot of the collection failed while it stayed live.
func TestOversizeRecordLeavesSnapshotWritable(t *testing.T) {
	c, err := NewCollection("default/snap", oversizeDenseConfig())
	if err != nil {
		t.Fatalf("NewCollection: %v", err)
	}
	vec := []float32{1, 0, 0, 0}
	wantTooLarge(t, "Insert", c.Insert(1, vec, 0, oversizeRecord(), nil))
	if err := c.Insert(2, vec, 0, Metadata{"session": NewRecord(sessionRecordBytes(t))}, nil); err != nil {
		t.Fatalf("normal Insert after a rejected one: %v", err)
	}
	var snap bytes.Buffer
	if err := c.Snapshot(&snap); err != nil {
		t.Fatalf("Snapshot: %v, want nil", err)
	}
	if snap.Len() == 0 {
		t.Fatal("Snapshot wrote no bytes")
	}
}
