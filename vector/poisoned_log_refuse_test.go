// SPDX-License-Identifier: Apache-2.0

package vector

import (
	"errors"
	"reflect"
	"testing"
	"time"
)

// Once a log is poisoned, every mutator of a log-backed collection refuses with
// ErrWALPoisoned BEFORE it applies anything: memory never moves ahead of the
// log, so a refused write is not visible until a restart, and a retried delete
// is refused again rather than reporting the point missing.

func recordStore(_ []byte, _ bool) ([]byte, RecordMutation, error) {
	return nil, RecordDelete, nil
}

type denseState struct {
	meta1    Metadata
	version1 uint64
	meta2    Metadata
	version2 uint64
	has9     bool
}

func denseStateOf(c *Collection) denseState {
	_, m1, _, _, v1, _ := c.Get(1)
	_, m2, _, _, v2, _ := c.Get(2)
	return denseState{meta1: m1, version1: v1, meta2: m2, version2: v2, has9: c.Exists(9)}
}

func TestPoisonedLogRefusesEveryDenseMutator(t *testing.T) {
	now := time.Now().UnixMilli()
	vec := durableVec(9)
	cases := []struct {
		name string
		op   func(c *Collection) error
	}{
		{"Insert", func(c *Collection) error { return c.Insert(9, vec, 0, nil, nil) }},
		{"InsertCASKeyTTLAt", func(c *Collection) error {
			_, err := c.InsertCASKeyTTLAt(9, vec, 0, nil, nil, nil, CASCond{}, now)
			return err
		}},
		{"RestoreInsert", func(c *Collection) error { return c.RestoreInsert(9, vec, 0, nil, nil, nil, 3) }},
		{"RestoreInsertAt", func(c *Collection) error { return c.RestoreInsertAt(9, vec, 0, nil, nil, nil, 3, now) }},
		{"InsertIfAbsentVersion", func(c *Collection) error {
			_, err := c.InsertIfAbsentVersion(9, vec, 0, nil, nil, nil, 0)
			return err
		}},
		{"InsertIfAbsentVersionAt", func(c *Collection) error {
			_, err := c.InsertIfAbsentVersionAt(9, vec, 0, nil, nil, nil, 0, now)
			return err
		}},
		{"Upsert over a live id", func(c *Collection) error { return c.Upsert(1, vec, "new", 0, nil, nil) }},
		{"UpsertCASKeyTTLAt", func(c *Collection) error {
			_, err := c.UpsertCASKeyTTLAt(1, vec, "new", 0, nil, nil, nil, CASCond{}, now)
			return err
		}},
		{"DeleteCAS of a live id", func(c *Collection) error { _, err := c.DeleteCAS(2, CASCond{}); return err }},
		{"DeleteCAS of an absent id", func(c *Collection) error { _, err := c.DeleteCAS(4242, CASCond{}); return err }},
		{"DeleteCASAt", func(c *Collection) error { _, err := c.DeleteCASAt(2, CASCond{}, now); return err }},
		{"SetPayloadCAS", func(c *Collection) error {
			_, err := c.SetPayloadCAS(1, Metadata{"k": NewInt(1)}, nil, CASCond{})
			return err
		}},
		{"OverwritePayloadCAS", func(c *Collection) error {
			_, err := c.OverwritePayloadCAS(1, Metadata{"k": NewInt(1)}, nil, CASCond{})
			return err
		}},
		{"DeletePayloadKeysCAS", func(c *Collection) error {
			_, err := c.DeletePayloadKeysCAS(1, []string{contentField}, CASCond{})
			return err
		}},
		{"ClearPayloadCAS", func(c *Collection) error { _, err := c.ClearPayloadCAS(1, CASCond{}); return err }},
		{"MutatePayloadRecordCAS", func(c *Collection) error {
			_, err := c.MutatePayloadRecordCAS(1, "rec", recordStore, CASCond{})
			return err
		}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			cs := openReplaceStore(t, t.TempDir(), durablePersistentCfg())
			t.Cleanup(func() { _ = cs.Close() })
			c := acquire(t, cs)
			defer c.Release()
			before := denseStateOf(c)
			c.wal.poisoned.Store(true)
			if err := tc.op(c); !errors.Is(err, ErrWALPoisoned) {
				t.Fatalf("%s on a poisoned log = %v, want ErrWALPoisoned", tc.name, err)
			}
			if after := denseStateOf(c); !reflect.DeepEqual(after, before) {
				t.Fatalf("a refused %s changed memory:\nbefore %+v\nafter  %+v", tc.name, before, after)
			}
		})
	}
}

// TestPoisonedLogRetriedDeleteIsRefused is the reported sequence: a delete
// whose fsync fails, then a retry. The retry must not report the point missing.
func TestPoisonedLogRetriedDeleteIsRefused(t *testing.T) {
	cs := openReplaceStore(t, t.TempDir(), durablePersistentCfg())
	t.Cleanup(func() { _ = cs.Close() })
	c := acquire(t, cs)
	defer c.Release()
	walSyncFails(t, c.wal)
	if _, err := c.DeleteCAS(2, CASCond{}); err == nil {
		t.Fatal("a delete whose fsync failed was acknowledged")
	}
	removed, err := c.DeleteCAS(2, CASCond{})
	if !errors.Is(err, ErrWALPoisoned) {
		t.Fatalf("retried delete = (%v, %v), want ErrWALPoisoned", removed, err)
	}
}

func TestPoisonedLogRefusesDenseBulkBuild(t *testing.T) {
	dir := t.TempDir()
	cs := openDurable(t, dir, "bulk")
	t.Cleanup(func() { _ = cs.Close() })
	if err := cs.CreateCollection("bulk", durablePersistentCfg()); err != nil {
		t.Fatal(err)
	}
	c, ok := cs.Acquire("bulk")
	if !ok {
		t.Fatal("collection missing")
	}
	defer c.Release()
	c.wal.poisoned.Store(true)
	if err := c.StageBulk([]uint64{1}, [][]float32{durableVec(1)}); err != nil {
		t.Fatal(err)
	}
	if err := c.BuildStaged(1); !errors.Is(err, ErrWALPoisoned) {
		t.Fatalf("BuildStaged on a poisoned log = %v, want ErrWALPoisoned", err)
	}
	if err := c.BuildConcurrent([]uint64{1}, [][]float32{durableVec(1)}, 1); !errors.Is(err, ErrWALPoisoned) {
		t.Fatalf("BuildConcurrent on a poisoned log = %v, want ErrWALPoisoned", err)
	}
	if c.Exists(1) {
		t.Fatal("a refused bulk build changed memory")
	}
}

type namedState struct {
	payload1 Metadata
	version1 uint64
	has9     bool
}

func namedStateOf(nc *NamedCollection) namedState {
	_, p, _, v, _ := nc.Get(1)
	_, _, _, _, ok := nc.Get(9)
	return namedState{payload1: p, version1: v, has9: ok}
}

func TestPoisonedLogRefusesEveryNamedMutator(t *testing.T) {
	now := time.Now().UnixMilli()
	vecs := map[string][]float32{"title": {0, 1, 0, 0}}
	cases := []struct {
		name string
		op   func(nc *NamedCollection) error
	}{
		{"InsertCAS", func(nc *NamedCollection) error { _, err := nc.InsertCAS(9, vecs, nil, 0, CASCond{}); return err }},
		{"InsertCAS over a live id", func(nc *NamedCollection) error {
			_, err := nc.InsertCAS(1, vecs, Metadata{"k": NewInt(5)}, 0, CASCond{})
			return err
		}},
		{"InsertCASKeyTTLAt", func(nc *NamedCollection) error {
			_, err := nc.InsertCASKeyTTLAt(9, vecs, nil, nil, 0, nil, CASCond{}, now)
			return err
		}},
		{"DeleteCAS of a live id", func(nc *NamedCollection) error { _, _, err := nc.DeleteCAS(1, CASCond{}); return err }},
		{"DeleteCAS of an absent id", func(nc *NamedCollection) error { _, _, err := nc.DeleteCAS(4242, CASCond{}); return err }},
		{"SetPayloadCAS", func(nc *NamedCollection) error {
			_, err := nc.SetPayloadCAS(1, Metadata{"k": NewInt(9)}, nil, CASCond{})
			return err
		}},
		{"OverwritePayloadCASAt", func(nc *NamedCollection) error {
			_, err := nc.OverwritePayloadCASAt(1, Metadata{"k": NewInt(9)}, nil, CASCond{}, now)
			return err
		}},
		{"DeletePayloadKeysCAS", func(nc *NamedCollection) error {
			_, err := nc.DeletePayloadKeysCAS(1, []string{"k"}, CASCond{})
			return err
		}},
		{"ClearPayloadCAS", func(nc *NamedCollection) error { _, err := nc.ClearPayloadCAS(1, CASCond{}); return err }},
		{"MutatePayloadRecordCAS", func(nc *NamedCollection) error {
			_, err := nc.MutatePayloadRecordCAS(1, "rec", recordStore, CASCond{})
			return err
		}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			nc := seedB2Named(t)
			before := namedStateOf(nc)
			nc.wal.poisoned.Store(true)
			if err := tc.op(nc); !errors.Is(err, ErrWALPoisoned) {
				t.Fatalf("%s on a poisoned log = %v, want ErrWALPoisoned", tc.name, err)
			}
			if after := namedStateOf(nc); !reflect.DeepEqual(after, before) {
				t.Fatalf("a refused %s changed memory:\nbefore %+v\nafter  %+v", tc.name, before, after)
			}
		})
	}
}

type mvState struct {
	payload1 Metadata
	version1 uint64
	has9     bool
}

func mvStateOf(m *MultiVectorIndex) mvState {
	_, p, v, _ := m.Get(1)
	return mvState{payload1: p, version1: v, has9: m.Exists(9)}
}

func TestPoisonedLogRefusesEveryMultiVectorMutator(t *testing.T) {
	now := time.Now().UnixMilli()
	toks := [][]float32{{0, 1, 0, 0}}
	cases := []struct {
		name string
		op   func(m *MultiVectorIndex) error
	}{
		{"AddCAS", func(m *MultiVectorIndex) error { _, err := m.AddCAS(9, toks, nil, CASCond{}); return err }},
		{"AddCAS over a live id", func(m *MultiVectorIndex) error {
			_, err := m.AddCAS(1, toks, Metadata{"k": NewInt(5)}, CASCond{})
			return err
		}},
		{"AddCASKeyTTLSparseAt", func(m *MultiVectorIndex) error {
			_, err := m.AddCASKeyTTLSparseAt(9, toks, nil, nil, nil, CASCond{}, now)
			return err
		}},
		{"AddIfAbsent", func(m *MultiVectorIndex) error { _, err := m.AddIfAbsent(9, toks, nil); return err }},
		{"MultiAddIfAbsentVersion", func(m *MultiVectorIndex) error {
			_, err := m.MultiAddIfAbsentVersion(9, toks, nil, nil, 3)
			return err
		}},
		{"MultiRestoreAdd", func(m *MultiVectorIndex) error { return m.MultiRestoreAdd(9, toks, nil, nil, 3) }},
		{"DeleteCAS of a live id", func(m *MultiVectorIndex) error { _, _, err := m.DeleteCAS(1, CASCond{}); return err }},
		{"DeleteCAS of an absent id", func(m *MultiVectorIndex) error { _, _, err := m.DeleteCAS(4242, CASCond{}); return err }},
		{"SetPayloadCAS", func(m *MultiVectorIndex) error {
			_, err := m.SetPayloadCAS(1, Metadata{"k": NewInt(9)}, nil, CASCond{})
			return err
		}},
		{"OverwritePayloadCASAt", func(m *MultiVectorIndex) error {
			_, err := m.OverwritePayloadCASAt(1, Metadata{"k": NewInt(9)}, nil, CASCond{}, now)
			return err
		}},
		{"DeletePayloadKeysCAS", func(m *MultiVectorIndex) error {
			_, err := m.DeletePayloadKeysCAS(1, []string{"k"}, CASCond{})
			return err
		}},
		{"ClearPayloadCAS", func(m *MultiVectorIndex) error { _, err := m.ClearPayloadCAS(1, CASCond{}); return err }},
		{"MutatePayloadRecordCAS", func(m *MultiVectorIndex) error {
			_, err := m.MutatePayloadRecordCAS(1, "rec", recordStore, CASCond{})
			return err
		}},
		{"MultiBulkBuild", func(m *MultiVectorIndex) error {
			_, err := m.MultiBulkBuild([]MultiScanRecord{{ID: 9, Tokens: toks}}, 1)
			return err
		}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			m := seedB2MV(t)
			before := mvStateOf(m)
			m.wal.poisoned.Store(true)
			if err := tc.op(m); !errors.Is(err, ErrWALPoisoned) {
				t.Fatalf("%s on a poisoned log = %v, want ErrWALPoisoned", tc.name, err)
			}
			if after := mvStateOf(m); !reflect.DeepEqual(after, before) {
				t.Fatalf("a refused %s changed memory:\nbefore %+v\nafter  %+v", tc.name, before, after)
			}
		})
	}
}
