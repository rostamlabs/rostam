// SPDX-License-Identifier: Apache-2.0

package vector

import (
	"bytes"
	"errors"
	"os"
	"testing"
	"time"
)

// ---------------------------------------------------------------------------
// B2 regression: the named/MV WAL wrapper's error returns
// ---------------------------------------------------------------------------
//
// logPayloadOp on the named and MV families returns (version, err) — the APPLIED
// version rides along even when only the WAL write failed — where the dense
// payloadOpCASChanged returns (0, err). That divergence is deliberate and
// pre-dates the record mutator (vector/named.go, vector/multivector.go: "the
// pre-split code returned the applied version alongside the append error").
// Splitting logPayloadOp into a changed-aware core must not quietly adopt the
// dense contract, so these two tests characterise the existing behaviour at BOTH
// error returns and are run before and after the refactor.

// walAppendFails poisons w so the next appendFramedStaged fails immediately with
// ErrWALPoisoned, without touching the file — the append-side error return.
func walAppendFails(w *wal) { w.poisoned.Store(true) }

// walSyncFails swaps w's file for the write end of a pipe: a small framed record
// still writes, but f.Sync() on a pipe fails with EINVAL, so the append succeeds
// and commitWaitStaged fails — the tail error return. The pipe's read end is
// held open for the test's lifetime so the write never draws EPIPE.
//
// CLEANUP ORDERING. t.Cleanup runs LIFO, and the store's own `_ = cs.Close()`
// cleanup is registered FIRST (by seedB2Named/seedB2MV), so this one runs
// BEFORE it: by the time Close reaches the WAL, w.f is a closed pipe and its
// final Sync fails. That is harmless here and deliberate — those callers discard
// Close's error, and the swapped-away real log file is closed by this cleanup
// rather than leaked. A future caller that ASSERTS on Close's error must swap
// w.f back instead of relying on this ordering.
func walSyncFails(t *testing.T, w *wal) {
	t.Helper()
	pr, pw, err := os.Pipe()
	if err != nil {
		t.Fatalf("os.Pipe: %v", err)
	}
	w.mu.Lock()
	old := w.f
	w.f = pw
	w.mu.Unlock()
	t.Cleanup(func() {
		_ = pw.Close()
		_ = pr.Close()
		_ = old.Close() // the real log file the store's Close will no longer reach
	})
}

// seedB2Named opens a WAL-mode named collection in the store holding point 1 at
// version 1, and returns it with the store closed on cleanup.
func seedB2Named(t *testing.T) *NamedCollection {
	t.Helper()
	cs, err := OpenCollectionStore(t.TempDir())
	if err != nil {
		t.Fatalf("OpenCollectionStore: %v", err)
	}
	t.Cleanup(func() { _ = cs.Close() })
	if err := cs.CreateCollection("named", namedWALConfig()); err != nil {
		t.Fatalf("CreateCollection: %v", err)
	}
	nc, ok := cs.GetNamed("named")
	if !ok {
		t.Fatal("named collection missing after create")
	}
	if nc.wal == nil {
		t.Fatal("WAL-mode named collection has nil wal")
	}
	v, err := nc.InsertCAS(1, map[string][]float32{"title": {1, 0, 0, 0}}, Metadata{"k": NewInt(1)}, 0, CASCond{})
	if err != nil || v != 1 {
		t.Fatalf("seed insert: v=%d err=%v (want 1, nil)", v, err)
	}
	return nc
}

// seedB2MV is seedB2Named for the multi-vector family.
func seedB2MV(t *testing.T) *MultiVectorIndex {
	t.Helper()
	cs, err := OpenCollectionStore(t.TempDir())
	if err != nil {
		t.Fatalf("OpenCollectionStore: %v", err)
	}
	t.Cleanup(func() { _ = cs.Close() })
	if err := cs.CreateMultiVector("mv", mvWALConfig()); err != nil {
		t.Fatalf("CreateMultiVector: %v", err)
	}
	idx, ok := cs.GetMultiVector("mv")
	if !ok {
		t.Fatal("MV collection missing after create")
	}
	if idx.wal == nil {
		t.Fatal("WAL-mode MV collection has nil wal")
	}
	v, err := idx.AddCAS(1, [][]float32{{1, 0, 0, 0}}, Metadata{"k": NewInt(1)}, CASCond{})
	if err != nil || v != 1 {
		t.Fatalf("seed add: v=%d err=%v (want 1, nil)", v, err)
	}
	return idx
}

func TestNamedSetPayloadReturnsVersionOnWALAppendError(t *testing.T) {
	t.Run("append fails", func(t *testing.T) {
		nc := seedB2Named(t)
		walAppendFails(nc.wal)
		version, err := nc.SetPayloadCAS(1, Metadata{"k": NewInt(9)}, nil, CASCond{})
		if err == nil {
			t.Fatal("SetPayloadCAS succeeded on a poisoned WAL")
		}
		if version != 2 {
			t.Fatalf("version = %d alongside the append error, want the APPLIED 2 (named returns the applied version, not 0)", version)
		}
	})

	t.Run("commit wait fails", func(t *testing.T) {
		nc := seedB2Named(t)
		walSyncFails(t, nc.wal)
		version, err := nc.SetPayloadCAS(1, Metadata{"k": NewInt(9)}, nil, CASCond{})
		if err == nil {
			t.Fatal("SetPayloadCAS succeeded with a failing fsync")
		}
		if version != 2 {
			t.Fatalf("version = %d alongside the commit-wait error, want the APPLIED 2", version)
		}
	})

	t.Run("engine error still returns 0", func(t *testing.T) {
		nc := seedB2Named(t)
		version, err := nc.SetPayloadCAS(404, Metadata{"k": NewInt(9)}, nil, CASCond{})
		if !errors.Is(err, ErrIDNotFound) {
			t.Fatalf("err = %v, want ErrIDNotFound", err)
		}
		if version != 0 {
			t.Fatalf("version = %d for an apply that never ran, want 0", version)
		}
	})
}

func TestMVSetPayloadReturnsVersionOnWALAppendError(t *testing.T) {
	t.Run("append fails", func(t *testing.T) {
		idx := seedB2MV(t)
		walAppendFails(idx.wal)
		version, err := idx.SetPayloadCAS(1, Metadata{"k": NewInt(9)}, nil, CASCond{})
		if err == nil {
			t.Fatal("SetPayloadCAS succeeded on a poisoned WAL")
		}
		if version != 2 {
			t.Fatalf("version = %d alongside the append error, want the APPLIED 2 (MV returns the applied version, not 0)", version)
		}
	})

	t.Run("commit wait fails", func(t *testing.T) {
		idx := seedB2MV(t)
		walSyncFails(t, idx.wal)
		version, err := idx.SetPayloadCAS(1, Metadata{"k": NewInt(9)}, nil, CASCond{})
		if err == nil {
			t.Fatal("SetPayloadCAS succeeded with a failing fsync")
		}
		if version != 2 {
			t.Fatalf("version = %d alongside the commit-wait error, want the APPLIED 2", version)
		}
	})

	t.Run("engine error still returns 0", func(t *testing.T) {
		idx := seedB2MV(t)
		version, err := idx.SetPayloadCAS(404, Metadata{"k": NewInt(9)}, nil, CASCond{})
		if !errors.Is(err, ErrIDNotFound) {
			t.Fatalf("err = %v, want ErrIDNotFound", err)
		}
		if version != 0 {
			t.Fatalf("version = %d for an apply that never ran, want 0", version)
		}
	})
}

// ---------------------------------------------------------------------------
// the shared table, on the named and multi-vector engines
// ---------------------------------------------------------------------------

// namedMutateHarness seeds a heap-only named collection holding point 1 with an
// rc=7 "session" record — the same state every other harness starts from.
func namedMutateHarness(t *testing.T) mutateHarness {
	t.Helper()
	nc := newTestNamed(t)
	nc.now = func() int64 { return mutateHarnessBase }
	h := mutateHarness{
		name:   "named",
		setNow: func(ms int64) { nc.now = func() int64 { return ms } },
		insert: func(t *testing.T, id uint64, meta Metadata) {
			t.Helper()
			if _, err := nc.InsertCAS(id, namedVecs(), meta, 0, CASCond{}); err != nil {
				t.Fatalf("named: InsertCAS(%d): %v", id, err)
			}
		},
		setPayload: func(t *testing.T, id uint64, meta Metadata, keyTTLMs map[string]int64) {
			t.Helper()
			if err := nc.SetPayload(id, meta, keyTTLMs); err != nil {
				t.Fatalf("named: SetPayload(%d): %v", id, err)
			}
		},
		mutate: func(id uint64, key string, fn RecordMutator, cas CASCond) (uint64, bool, error) {
			_, _, v, changed, err := nc.mutatePayloadRecordLockedAt(id, key, fn, cas, nc.nowMs(), false)
			return v, changed, err
		},
		mutateAt: func(id uint64, key string, fn RecordMutator, cas CASCond, nowMs int64) (uint64, bool, error) {
			_, _, v, changed, err := nc.mutatePayloadRecordLockedAt(id, key, fn, cas, nowMs, true)
			return v, changed, err
		},
		mutateAtKE: func(id uint64, key string, fn RecordMutator, cas CASCond, nowMs int64) (map[string]int64, uint64, bool, error) {
			_, ke, v, changed, err := nc.mutatePayloadRecordLockedAt(id, key, fn, cas, nowMs, true)
			return ke, v, changed, err
		},
		payload: func(id uint64) (Metadata, bool) {
			_, meta, _, _, ok := nc.Get(id)
			return meta, ok
		},
		version: func(id uint64) uint64 {
			_, _, _, v, ok := nc.Get(id)
			if !ok {
				return 0
			}
			return v
		},
		stored: func(t *testing.T, id uint64) Metadata {
			t.Helper()
			nc.mu.RLock()
			defer nc.mu.RUnlock()
			return nc.meta[id]
		},
	}
	h.insert(t, 1, Metadata{"session": NewRecord(sessionRecordBytes(t))})
	return h
}

// mvMutateHarness is namedMutateHarness for the multi-vector family.
func mvMutateHarness(t *testing.T) mutateHarness {
	t.Helper()
	m := newTestMV(t)
	m.now = func() int64 { return mutateHarnessBase }
	h := mutateHarness{
		name:   "mv",
		setNow: func(ms int64) { m.now = func() int64 { return ms } },
		insert: func(t *testing.T, id uint64, meta Metadata) {
			t.Helper()
			if _, err := m.AddCAS(id, mvTokensV(), meta, CASCond{}); err != nil {
				t.Fatalf("mv: AddCAS(%d): %v", id, err)
			}
		},
		setPayload: func(t *testing.T, id uint64, meta Metadata, keyTTLMs map[string]int64) {
			t.Helper()
			if err := m.SetPayload(id, meta, keyTTLMs); err != nil {
				t.Fatalf("mv: SetPayload(%d): %v", id, err)
			}
		},
		mutate: func(id uint64, key string, fn RecordMutator, cas CASCond) (uint64, bool, error) {
			_, _, v, changed, err := m.mutatePayloadRecordLockedAt(id, key, fn, cas, m.nowMs(), false)
			return v, changed, err
		},
		mutateAt: func(id uint64, key string, fn RecordMutator, cas CASCond, nowMs int64) (uint64, bool, error) {
			_, _, v, changed, err := m.mutatePayloadRecordLockedAt(id, key, fn, cas, nowMs, true)
			return v, changed, err
		},
		mutateAtKE: func(id uint64, key string, fn RecordMutator, cas CASCond, nowMs int64) (map[string]int64, uint64, bool, error) {
			_, ke, v, changed, err := m.mutatePayloadRecordLockedAt(id, key, fn, cas, nowMs, true)
			return ke, v, changed, err
		},
		payload: func(id uint64) (Metadata, bool) {
			_, meta, _, ok := m.Get(id)
			return meta, ok
		},
		version: func(id uint64) uint64 {
			_, _, v, ok := m.Get(id)
			if !ok {
				return 0
			}
			return v
		},
		stored: func(t *testing.T, id uint64) Metadata {
			t.Helper()
			m.mu.RLock()
			defer m.mu.RUnlock()
			return m.docMeta[id]
		},
	}
	h.insert(t, 1, Metadata{"session": NewRecord(sessionRecordBytes(t))})
	return h
}

// allMutateHarnesses builds one harness per engine. Every engine carries its own
// copy of the record-mutation body, so they are all driven from the SAME table:
// a divergence in any one of the four fails here.
func allMutateHarnesses(t *testing.T) []func(*testing.T) mutateHarness {
	t.Helper()
	return []func(*testing.T) mutateHarness{
		denseMutateHarness, ivfMutateHarness, namedMutateHarness, mvMutateHarness,
	}
}

func TestMutateRecordNamedAndMVMatchDense(t *testing.T) {
	t.Run("shared table", func(t *testing.T) {
		for _, tc := range mutateCases(t) {
			t.Run(tc.name, func(t *testing.T) {
				for _, newH := range allMutateHarnesses(t) {
					h := newH(t)
					t.Run(h.name, func(t *testing.T) { runMutateCase(t, h, tc) })
				}
			})
		}
	})

	t.Run("creates under an absent key", func(t *testing.T) {
		for _, newH := range allMutateHarnesses(t) {
			h := newH(t)
			t.Run(h.name, func(t *testing.T) {
				want := sessionRecordBytesRC(t, 8)
				var sawOld []byte
				var sawExists, called bool
				version, changed, err := h.mutate(1, "fresh", func(old []byte, exists bool) ([]byte, RecordMutation, error) {
					called, sawOld, sawExists = true, old, exists
					return want, RecordStore, nil
				}, CASCond{})
				if err != nil {
					t.Fatalf("mutate: %v", err)
				}
				if !called {
					t.Fatal("the mutator was never called for an absent key")
				}
				if sawOld != nil || sawExists {
					t.Errorf("the mutator saw (%v, %v) for an absent key, want (nil, false)", sawOld, sawExists)
				}
				if !changed || version != 2 {
					t.Fatalf("changed=%v version=%d, want true/2", changed, version)
				}
				if got, ok := h.record(t, 1, "fresh"); !ok || !bytes.Equal(got, want) {
					t.Fatal("the new key does not hold the stored record")
				}
				if got, ok := h.record(t, 1, "session"); !ok || !bytes.Equal(got, sessionRecordBytes(t)) {
					t.Fatal("the untouched key changed — a mutation must only touch its own key")
				}
			})
		}
	})

	t.Run("non-record value is an error", func(t *testing.T) {
		for _, newH := range allMutateHarnesses(t) {
			h := newH(t)
			t.Run(h.name, func(t *testing.T) {
				h.insert(t, 2, Metadata{"session": NewRecord(sessionRecordBytes(t)), "country": NewString("DE")})
				before := h.version(2)
				var called bool
				_, changed, err := h.mutate(2, "country", func(_ []byte, _ bool) ([]byte, RecordMutation, error) {
					called = true
					return sessionRecordBytes(t), RecordStore, nil
				}, CASCond{})
				if !errors.Is(err, ErrPayloadKeyNotRecord) {
					t.Fatalf("err = %v, want ErrPayloadKeyNotRecord", err)
				}
				if called {
					t.Error("the mutator ran against a non-record value")
				}
				if changed {
					t.Error("changed=true on a rejected call")
				}
				meta, ok := h.payload(2)
				if !ok {
					t.Fatal("point 2 is not live")
				}
				if got := meta["country"]; !got.Equal(NewString("DE")) {
					t.Errorf("the non-record value changed: %+v", got)
				}
				if v := h.version(2); v != before {
					t.Errorf("version = %d after a rejected call, want %d", v, before)
				}
			})
		}
	})

	t.Run("absent id is not found", func(t *testing.T) {
		for _, newH := range allMutateHarnesses(t) {
			h := newH(t)
			t.Run(h.name, func(t *testing.T) {
				var called bool
				version, changed, err := h.mutate(404, "session", func(_ []byte, _ bool) ([]byte, RecordMutation, error) {
					called = true
					return sessionRecordBytesRC(t, 8), RecordStore, nil
				}, CASCond{})
				if !errors.Is(err, ErrIDNotFound) {
					t.Fatalf("err = %v, want ErrIDNotFound", err)
				}
				if called {
					t.Error("the mutator ran against an absent point")
				}
				if changed || version != 0 {
					t.Errorf("changed=%v version=%d against an absent point, want false/0", changed, version)
				}
			})
		}
	})

	// The stamped clock: the per-key deadline check AND the stale-deadline drop
	// must come from the explicit stamp, never from the engine's wall clock, or
	// two replicas applying the same op would store different bytes.
	t.Run("stamped clock decides expiry and drops the stale deadline", func(t *testing.T) {
		for _, newH := range allMutateHarnesses(t) {
			h := newH(t)
			t.Run(h.name, func(t *testing.T) {
				// Pin the wall clock at 4000 so the seeding set_payload gives
				// "session" an ABSOLUTE deadline of 5000, then move it far away.
				h.setNow(4000)
				h.setPayload(t, 1, Metadata{"session": NewRecord(sessionRecordBytes(t))},
					map[string]int64{"session": 1000})
				h.setNow(9_000_000) // far past every stamp used below

				// One millisecond before the deadline the key is still live.
				var sawExists bool
				if _, _, err := h.mutateAt(1, "session", func(_ []byte, exists bool) ([]byte, RecordMutation, error) {
					sawExists = exists
					return nil, RecordUnchanged, nil
				}, CASCond{}, 4999); err != nil {
					t.Fatalf("mutateAt(4999): %v", err)
				}
				if !sawExists {
					t.Fatal("at stamp 4999 the mutator saw the key as absent — the wall clock was consulted")
				}

				// At the deadline the key reads as ABSENT, and storing under it
				// must drop the stale deadline, or the record just written would
				// be invisible the instant it lands.
				want := sessionRecordBytesRC(t, 8)
				_, changed, err := h.mutateAt(1, "session", func(_ []byte, exists bool) ([]byte, RecordMutation, error) {
					sawExists = exists
					return want, RecordStore, nil
				}, CASCond{}, 5000)
				if err != nil {
					t.Fatalf("mutateAt(5000): %v", err)
				}
				if sawExists {
					t.Fatal("at stamp 5000 the mutator saw the expired key as present")
				}
				if !changed {
					t.Fatal("changed=false storing under an expired key")
				}
				h.setNow(6000)
				got, ok := h.record(t, 1, "session")
				if !ok || !bytes.Equal(got, want) {
					t.Fatal("the freshly stored record is invisible at now=6000 — the stale deadline was not dropped")
				}
			})
		}
	})

	// A mutation touches ONE key. The engines rebuild the whole per-key deadline
	// map when they write, so the hazard is not that the other key's VALUE
	// changes — the "creates under an absent key" case already covers that — but
	// that its DEADLINE is dropped or recomputed on the way through, silently
	// making a key with hours left expire now or never.
	t.Run("a mutation preserves the other keys' deadlines", func(t *testing.T) {
		for _, newH := range allMutateHarnesses(t) {
			h := newH(t)
			t.Run(h.name, func(t *testing.T) {
				// At now=4000: "session" gets deadline 5000, "other" gets 9000.
				h.setNow(4000)
				h.setPayload(t, 1, Metadata{
					"session": NewRecord(sessionRecordBytes(t)),
					"other":   NewString("keep me"),
				}, map[string]int64{"session": 1000, "other": 5000})

				ke, _, changed, err := h.mutateAtKE(1, "session",
					storeFn(sessionRecordBytesRC(t, 8)), CASCond{}, 4500)
				if err != nil {
					t.Fatalf("mutateAtKE(4500): %v", err)
				}
				if !changed {
					t.Fatal("changed=false storing under a live key")
				}
				if got := ke["other"]; got != 9000 {
					t.Errorf("the untouched key's deadline = %d, want 9000 (deadlines: %v)", got, ke)
				}

				// And it is a real deadline, not just a number in the returned
				// map: the key is live before 9000 and gone after it.
				h.setNow(8999)
				if meta, ok := h.payload(1); !ok || !meta["other"].Equal(NewString("keep me")) {
					t.Errorf("the untouched key is not live at now=8999: %v", meta)
				}
				h.setNow(9000)
				if meta, ok := h.payload(1); ok {
					if _, present := meta["other"]; present {
						t.Errorf("the untouched key outlived its deadline at now=9000: %v", meta)
					}
				}
			})
		}
	})
}

// TestMutateRecordNamedPointTTLExpiryGate is the named family's POINT-ttl gate,
// the twin of the dense TestMutateRecordDeadPointNotFound. A point whose own TTL
// has passed is gone: the mutation must report ErrIDNotFound against the STAMPED
// clock, and — because a mutator is caller code that may have side effects — it
// must never run at all.
func TestMutateRecordNamedPointTTLExpiryGate(t *testing.T) {
	nc := newTestNamed(t)
	nc.now = func() int64 { return mutateHarnessBase }
	// A 1s point TTL: the deadline is mutateHarnessBase+1000.
	if _, err := nc.InsertCAS(1, namedVecs(),
		Metadata{"session": NewRecord(sessionRecordBytes(t))}, time.Second, CASCond{}); err != nil {
		t.Fatalf("InsertCAS: %v", err)
	}
	// The wall clock is moved far past the deadline, so any judgement below that
	// comes out "live" proves the stamp — not the clock — decided it.
	nc.now = func() int64 { return mutateHarnessBase + 9_000_000 }

	// One millisecond before the deadline the point is still there.
	var called bool
	if _, _, _, changed, err := nc.mutatePayloadRecordLockedAt(1, "session",
		func(_ []byte, _ bool) ([]byte, RecordMutation, error) {
			called = true
			return nil, RecordUnchanged, nil
		}, CASCond{}, mutateHarnessBase+999, true); err != nil {
		t.Fatalf("mutate at stamp base+999: %v, want nil", err)
	} else if changed {
		t.Error("changed=true for RecordUnchanged")
	}
	if !called {
		t.Fatal("the mutator never ran against a live point — the wall clock was consulted")
	}

	// At the deadline it is gone.
	called = false
	version, changed, err := func() (uint64, bool, error) {
		_, _, v, ch, e := nc.mutatePayloadRecordLockedAt(1, "session",
			func(_ []byte, _ bool) ([]byte, RecordMutation, error) {
				called = true
				return sessionRecordBytesRC(t, 8), RecordStore, nil
			}, CASCond{}, mutateHarnessBase+1000, true)
		return v, ch, e
	}()
	if !errors.Is(err, ErrIDNotFound) {
		t.Fatalf("err = %v, want ErrIDNotFound", err)
	}
	if called {
		t.Error("the mutator ran against a point past its TTL")
	}
	if changed || version != 0 {
		t.Errorf("changed=%v version=%d against an expired point, want false/0", changed, version)
	}
}

// ---------------------------------------------------------------------------
// named: the shared payload's id-keyed index after a mutation
// ---------------------------------------------------------------------------

// recordIDIn reports whether the id-keyed index posts id under field=key.
func recordIDIn(p *payloadIndexID, field string, key scalarKey, id uint64) bool {
	vals := p.fields[field]
	if vals == nil {
		return false
	}
	set := vals[key]
	if set == nil {
		return false
	}
	_, ok := set[id]
	return ok
}

// assertRecordPostingsExactID is assertRecordPostingsExact for the id-keyed
// mirror: for the synthetic field, the index's postings agree EXACTLY with what
// lookupPath resolves from each id's stored payload (the map reindex was last
// handed).
func assertRecordPostingsExactID(t *testing.T, p *payloadIndexID, meta map[uint64]Metadata, ids []uint64, field string) {
	t.Helper()
	posted := make(map[uint64]scalarKey)
	for key, set := range p.fields[field] {
		for id := range set {
			if prev, dup := posted[id]; dup {
				t.Errorf("id %d is posted under two keys for %q: %+v and %+v", id, field, prev, key)
			}
			posted[id] = key
		}
	}
	for _, id := range ids {
		v, resolved := lookupPath(meta[id], field)
		key, scalar := scalarKeyOf(v)
		gotKey, gotPosted := posted[id]
		delete(posted, id)
		if resolved && scalar {
			if !gotPosted {
				t.Errorf("id %d resolves %q = %+v but has no posting", id, field, key)
			} else if gotKey != key {
				t.Errorf("id %d resolves %q = %+v but is posted under %+v", id, field, key, gotKey)
			}
			continue
		}
		if gotPosted {
			t.Errorf("id %d does not resolve %q but is posted under %+v", id, field, gotKey)
		}
	}
	for id, key := range posted {
		t.Errorf("%q has a posting for id %d under %+v that no live id explains", field, id, key)
	}
}

func TestMutateRecordNamedSharedPayloadReindex(t *testing.T) {
	t.Run("postings stay exact", func(t *testing.T) {
		nc := newTestNamed(t)
		ids := make([]uint64, 0, 12)
		for i := 1; i <= 12; i++ {
			id := uint64(i)
			meta := Metadata{"session": NewRecord(sessionRecordBytesRC(t, uint8(i%5)))} //nolint:gosec // bounded
			if _, err := nc.InsertCAS(id, namedVecs(), meta, 0, CASCond{}); err != nil {
				t.Fatalf("InsertCAS(%d): %v", id, err)
			}
			ids = append(ids, id)
		}
		// Point 1 goes to rc=7 and then to rc=8, so the mutation must move it.
		if _, err := nc.MutatePayloadRecordCAS(1, "session", storeFn(sessionRecordBytesRC(t, 7)), CASCond{}); err != nil {
			t.Fatalf("MutatePayloadRecordCAS: %v", err)
		}
		nc.mu.RLock()
		assertRecordPostingsExactID(t, nc.payloadIdx, nc.meta, ids, "session/rc")
		nc.mu.RUnlock()

		if _, err := nc.MutatePayloadRecordCAS(1, "session", storeFn(sessionRecordBytesRC(t, 8)), CASCond{}); err != nil {
			t.Fatalf("MutatePayloadRecordCAS: %v", err)
		}
		nc.mu.RLock()
		defer nc.mu.RUnlock()
		if recordIDIn(nc.payloadIdx, "session/rc", intKey(7), 1) {
			t.Error("fields[session/rc][7] still holds the mutated id — stale posting after a record mutation")
		}
		if !recordIDIn(nc.payloadIdx, "session/rc", intKey(8), 1) {
			t.Error("fields[session/rc][8] is missing the mutated id")
		}
		assertRecordPostingsExactID(t, nc.payloadIdx, nc.meta, ids, "session/rc")
	})

	t.Run("bad-record poison clears", func(t *testing.T) {
		nc := newTestNamed(t)
		// A mode byte followed by a torn schema blob: IndexEntries cannot
		// enumerate it, so the payload key is poisoned and nothing under it
		// accelerates. The accounting is per-ID here, not per-slot.
		//
		// Seeded through NamedCollection.RestoreInsert, the named family's REPLAY
		// entry (size-checked, not shape-checked — see checkRecordValuesSize):
		// InsertCAS now refuses a malformed record with ErrRecordMalformed, so
		// this state is only reachable the way it originally arose, from bytes
		// stored before the gate existed. version 0 bumps a fresh point to 1
		// exactly as InsertCAS did.
		if err := nc.RestoreInsert(1, namedVecs(), nil, Metadata{"session": NewRecord([]byte{0x01, 0xFF})}, 0, nil, 0); err != nil {
			t.Fatalf("RestoreInsert: %v", err)
		}
		if !nc.payloadIdx.badRecords.poisoned("session") {
			t.Fatal("the malformed record did not poison the payload key — the fixture is wrong")
		}
		want := sessionRecordBytes(t)
		if _, err := nc.MutatePayloadRecordCAS(1, "session", storeFn(want), CASCond{}); err != nil {
			t.Fatalf("MutatePayloadRecordCAS over a malformed record: %v", err)
		}
		nc.mu.RLock()
		defer nc.mu.RUnlock()
		if nc.payloadIdx.badRecords.poisoned("session") {
			t.Fatal("the payload key stayed poisoned after the malformed record was replaced")
		}
		if !recordIDIn(nc.payloadIdx, "session/rc", intKey(7), 1) {
			t.Error("the repaired record did not regain acceleration (no session/rc posting)")
		}
	})
}

// ---------------------------------------------------------------------------
// named/MV WAL: one record per applied mutation, replayed verbatim
// ---------------------------------------------------------------------------

// walMutateEngine is one WAL-mode family behind the reopen table.
type walMutateEngine struct {
	mutate   func(key string, fn RecordMutator, cas CASCond) (uint64, error)
	mutateAt func(key string, fn RecordMutator, cas CASCond, nowMs int64) (uint64, error)
	payload  func() (Metadata, bool)
	version  func() uint64
	stored   func(t *testing.T) Metadata
	postings func() map[scalarKey]map[uint64]struct{} // session/rc postings
	wal      *wal
}

func namedWALEngine(nc *NamedCollection) walMutateEngine {
	return walMutateEngine{
		mutate: func(key string, fn RecordMutator, cas CASCond) (uint64, error) {
			return nc.MutatePayloadRecordCAS(1, key, fn, cas)
		},
		mutateAt: func(key string, fn RecordMutator, cas CASCond, nowMs int64) (uint64, error) {
			return nc.MutatePayloadRecordCASAt(1, key, fn, cas, nowMs)
		},
		payload: func() (Metadata, bool) {
			_, meta, _, _, ok := nc.Get(1)
			return meta, ok
		},
		version: func() uint64 {
			_, _, _, v, ok := nc.Get(1)
			if !ok {
				return 0
			}
			return v
		},
		stored: func(t *testing.T) Metadata {
			t.Helper()
			nc.mu.RLock()
			defer nc.mu.RUnlock()
			return nc.meta[1]
		},
		postings: func() map[scalarKey]map[uint64]struct{} {
			nc.mu.RLock()
			defer nc.mu.RUnlock()
			return nc.payloadIdx.fields["session/rc"]
		},
		wal: nc.wal,
	}
}

func mvWALEngine(m *MultiVectorIndex) walMutateEngine {
	return walMutateEngine{
		mutate: func(key string, fn RecordMutator, cas CASCond) (uint64, error) {
			return m.MutatePayloadRecordCAS(1, key, fn, cas)
		},
		mutateAt: func(key string, fn RecordMutator, cas CASCond, nowMs int64) (uint64, error) {
			return m.MutatePayloadRecordCASAt(1, key, fn, cas, nowMs)
		},
		payload: func() (Metadata, bool) {
			_, meta, _, ok := m.Get(1)
			return meta, ok
		},
		version: func() uint64 {
			_, _, v, ok := m.Get(1)
			if !ok {
				return 0
			}
			return v
		},
		stored: func(t *testing.T) Metadata {
			t.Helper()
			m.mu.RLock()
			defer m.mu.RUnlock()
			return m.docMeta[1]
		},
		postings: func() map[scalarKey]map[uint64]struct{} {
			m.mu.RLock()
			defer m.mu.RUnlock()
			return m.payloadIdx.fields["session/rc"]
		},
		wal: m.wal,
	}
}

// walMutateFamily opens (or reopens) a WAL-mode collection of one family in dir,
// holding point/doc 1 with an rc=7 "session" record. The caller owns Close — the
// reopen cases close deliberately, without a Flush.
type walMutateFamily struct {
	name string
	open func(t *testing.T, dir string) (*CollectionStore, walMutateEngine)
}

func walMutateFamilies() []walMutateFamily {
	return []walMutateFamily{
		{
			name: "named",
			open: func(t *testing.T, dir string) (*CollectionStore, walMutateEngine) {
				t.Helper()
				cs, err := OpenCollectionStore(dir)
				if err != nil {
					t.Fatalf("OpenCollectionStore: %v", err)
				}
				nc, ok := cs.GetNamed("named")
				if !ok {
					if err := cs.CreateCollection("named", namedWALConfig()); err != nil {
						t.Fatalf("CreateCollection: %v", err)
					}
					if nc, ok = cs.GetNamed("named"); !ok {
						t.Fatal("named collection missing after create")
					}
					if _, err := nc.InsertCAS(1, map[string][]float32{"title": {1, 0, 0, 0}},
						Metadata{"session": NewRecord(sessionRecordBytes(t))}, 0, CASCond{}); err != nil {
						t.Fatalf("seed insert: %v", err)
					}
				}
				if nc.wal == nil {
					t.Fatal("WAL-mode named collection has nil wal")
				}
				return cs, namedWALEngine(nc)
			},
		},
		{
			name: "mv",
			open: func(t *testing.T, dir string) (*CollectionStore, walMutateEngine) {
				t.Helper()
				cs, err := OpenCollectionStore(dir)
				if err != nil {
					t.Fatalf("OpenCollectionStore: %v", err)
				}
				idx, ok := cs.GetMultiVector("mv")
				if !ok {
					if err := cs.CreateMultiVector("mv", mvWALConfig()); err != nil {
						t.Fatalf("CreateMultiVector: %v", err)
					}
					if idx, ok = cs.GetMultiVector("mv"); !ok {
						t.Fatal("MV collection missing after create")
					}
					if _, err := idx.AddCAS(1, [][]float32{{1, 0, 0, 0}, {0, 1, 0, 0}},
						Metadata{"session": NewRecord(sessionRecordBytes(t))}, CASCond{}); err != nil {
						t.Fatalf("seed add: %v", err)
					}
				}
				if idx.wal == nil {
					t.Fatal("WAL-mode MV collection has nil wal")
				}
				return cs, mvWALEngine(idx)
			},
		},
	}
}

func TestMutateRecordNamedAndMVSurviveReopen(t *testing.T) {
	const stamp int64 = 1_700_000_000_000
	cases := []struct {
		name   string
		mutate func(t *testing.T, e walMutateEngine) (version uint64)
		verify func(t *testing.T, e walMutateEngine, version uint64)
	}{
		{
			name: "store",
			mutate: func(t *testing.T, e walMutateEngine) uint64 {
				v, err := e.mutate("session", storeFn(sessionRecordBytesRC(t, 8)), CASCond{})
				if err != nil {
					t.Fatalf("MutatePayloadRecordCAS: %v", err)
				}
				return v
			},
			verify: func(t *testing.T, e walMutateEngine, version uint64) {
				meta, ok := e.payload()
				if !ok {
					t.Fatal("point 1 lost across reopen")
				}
				if v := e.version(); v != version {
					t.Fatalf("version = %d after replay, want the acked %d", v, version)
				}
				if got := meta["session"]; got.Kind != ValueRecord || !bytes.Equal(got.Rec, sessionRecordBytesRC(t, 8)) {
					t.Fatal("the mutated record did not replay byte-identically")
				}
			},
		},
		{
			name: "delete",
			mutate: func(t *testing.T, e walMutateEngine) uint64 {
				v, err := e.mutate("session",
					func(_ []byte, _ bool) ([]byte, RecordMutation, error) { return nil, RecordDelete, nil }, CASCond{})
				if err != nil {
					t.Fatalf("MutatePayloadRecordCAS (delete): %v", err)
				}
				return v
			},
			verify: func(t *testing.T, e walMutateEngine, version uint64) {
				meta, ok := e.payload()
				if !ok {
					t.Fatal("point 1 lost across reopen")
				}
				if v := e.version(); v != version {
					t.Fatalf("version = %d after replay, want the acked %d", v, version)
				}
				if _, present := meta["session"]; present {
					t.Fatal("the deleted key came back on replay — the WAL record is a REPLACE, not a merge")
				}
				if got := e.stored(t); got != nil {
					t.Errorf("replayed stored payload = %v, want nil (the pre-restart state)", got)
				}
				if p := e.postings(); len(p) != 0 {
					t.Fatalf("the rebuilt index still posts session/rc: %v", p)
				}
			},
		},
		{
			name: "stamped op-list",
			mutate: func(t *testing.T, e walMutateEngine) uint64 {
				v, err := e.mutateAt("session", storeFn(stampedRecordBytes(t, stamp)), CASCond{}, stamp)
				if err != nil {
					t.Fatalf("MutatePayloadRecordCASAt: %v", err)
				}
				return v
			},
			verify: func(t *testing.T, e walMutateEngine, version uint64) {
				meta, ok := e.payload()
				if !ok {
					t.Fatal("point 1 lost across reopen")
				}
				if v := e.version(); v != version {
					t.Fatalf("version = %d after replay, want the acked %d", v, version)
				}
				if got := meta["session"]; !bytes.Equal(got.Rec, stampedRecordBytes(t, stamp)) {
					t.Fatal("the stamped record changed on replay — the op-list must never be re-run")
				}
			},
		},
	}

	for _, fam := range walMutateFamilies() {
		t.Run(fam.name, func(t *testing.T) {
			for _, tc := range cases {
				t.Run(tc.name, func(t *testing.T) {
					dir := t.TempDir()
					cs, e := fam.open(t, dir)
					version := tc.mutate(t, e)
					// NO Flush: the WAL alone must carry the mutation.
					if err := cs.Close(); err != nil {
						t.Fatalf("Close: %v", err)
					}
					reopened, e2 := fam.open(t, dir)
					t.Cleanup(func() { _ = reopened.Close() })
					tc.verify(t, e2, version)
				})
			}
		})
	}
}

// walSeqOf reads a WAL's monotonic record counter under its own lock.
func walSeqOf(w *wal) uint64 {
	w.syncMu.Lock()
	defer w.syncMu.Unlock()
	return w.writeSeq
}

// walSizeOf reads the WAL file's on-disk size.
func walSizeOf(t *testing.T, w *wal) int64 {
	t.Helper()
	w.mu.Lock()
	defer w.mu.Unlock()
	fi, err := w.f.Stat()
	if err != nil {
		t.Fatalf("stat wal: %v", err)
	}
	return fi.Size()
}

// TestMutateRecordNamedWALNoOp pins the whole point of the changed flag on the
// named and MV wrappers: a mutation that reports RecordUnchanged (a failed CHECK)
// stages NOTHING — neither a log record nor a byte on disk — while a RecordStore
// stages exactly one.
func TestMutateRecordNamedWALNoOp(t *testing.T) {
	for _, fam := range walMutateFamilies() {
		t.Run(fam.name, func(t *testing.T) {
			cs, e := fam.open(t, t.TempDir())
			t.Cleanup(func() { _ = cs.Close() })

			before := e.version()
			seqBefore, sizeBefore := walSeqOf(e.wal), walSizeOf(t, e.wal)
			version, err := e.mutate("session",
				func(_ []byte, _ bool) ([]byte, RecordMutation, error) { return nil, RecordUnchanged, nil }, CASCond{})
			if err != nil {
				t.Fatalf("MutatePayloadRecordCAS: %v", err)
			}
			if version != before {
				t.Fatalf("version = %d after a no-op, want the unbumped %d", version, before)
			}
			if got := walSeqOf(e.wal); got != seqBefore {
				t.Fatalf("the WAL advanced %d -> %d on a no-op — a failed CHECK must not cost a log record", seqBefore, got)
			}
			if got := walSizeOf(t, e.wal); got != sizeBefore {
				t.Fatalf("the WAL file grew %d -> %d bytes on a no-op", sizeBefore, got)
			}

			// A real store stages exactly one record.
			if _, err := e.mutate("session", storeFn(sessionRecordBytesRC(t, 8)), CASCond{}); err != nil {
				t.Fatalf("MutatePayloadRecordCAS (store): %v", err)
			}
			if got := walSeqOf(e.wal); got != seqBefore+1 {
				t.Fatalf("the WAL advanced %d -> %d on a store, want exactly one record", seqBefore, got)
			}
			if e.version() != before+1 {
				t.Fatalf("version = %d after a store, want %d", e.version(), before+1)
			}
		})
	}
}

// TestMutateRejectsMalformedMutatorBytes drives every engine's post-mutation
// SHAPE gate, the twin of the size cap beside it.
//
// MutatePayloadRecordCAS is exported. The mutator ops supplies cannot produce
// malformed bytes — it returns applyRecordBytes' output — but a direct Go caller
// brings its own, and bytes that no operate engine can open used to be stored on
// the size cap alone. That poisons the payload key for the whole collection: no
// path under it accelerates, and the next vector_operate on the same key fails.
//
// The refusal must leave the point byte-identical AND unbumped, exactly like a
// failed CHECK — the record the caller already had is not collateral for a bad
// mutator.
func TestMutateRejectsMalformedMutatorBytes(t *testing.T) {
	garbage := []byte{0xff, 0x00, 0x13, 0x37}
	if err := validateRecord(garbage); err == nil {
		t.Fatal("fixture is not malformed: validateRecord accepted it")
	}
	for _, newH := range allMutateHarnesses(t) {
		h := newH(t)
		t.Run(h.name, func(t *testing.T) {
			before, ok := h.record(t, 1, "session")
			if !ok {
				t.Fatal("harness seed is missing the session record")
			}
			beforeVersion := h.version(1)

			_, changed, err := h.mutate(1, "session", func(_ []byte, _ bool) ([]byte, RecordMutation, error) {
				return garbage, RecordStore, nil
			}, CASCond{})
			if !errors.Is(err, ErrRecordMalformed) {
				t.Fatalf("a mutator returning malformed bytes = %v, want ErrRecordMalformed", err)
			}
			if changed {
				t.Fatal("the refused mutation reported changed=true")
			}
			if !IsRecordMalformedMessage(err.Error()) {
				t.Fatalf("message %q is not the shape the transport classifiers match; "+
					"a clustered caller would see it redacted to an internal error", err)
			}

			after, ok := h.record(t, 1, "session")
			if !ok {
				t.Fatal("the refused mutation removed the record")
			}
			if !bytes.Equal(before, after) {
				t.Fatal("the refused mutation changed the stored record")
			}
			if got := h.version(1); got != beforeVersion {
				t.Fatalf("version = %d after a refused mutation, want %d (unbumped)", got, beforeVersion)
			}
		})
	}
}

// TestMutateValidatesRecordExactlyOnce pins the cost of the gate above: one
// decode of the bytes the engine just produced, never two. The shape check is
// O(record) with allocations, so a path that grew a second pass would multiply
// the cost of every operate — the same invariant
// TestRecordValidationCountPerWrite holds for the ingest paths.
func TestMutateValidatesRecordExactlyOnce(t *testing.T) {
	want := sessionRecordBytesRC(t, 9)
	for _, newH := range allMutateHarnesses(t) {
		h := newH(t)
		t.Run(h.name, func(t *testing.T) {
			n := countValidates(t, func() {
				if _, _, err := h.mutate(1, "session", func(_ []byte, _ bool) ([]byte, RecordMutation, error) {
					return want, RecordStore, nil
				}, CASCond{}); err != nil {
					t.Fatalf("mutate: %v", err)
				}
			})
			if n != 1 {
				t.Fatalf("a record mutation decoded its result %d times, want exactly 1", n)
			}
		})
	}
}

// TestMutateSkipsValidationWhenNothingIsStored is the other half of the cost
// invariant: a mutation that stores nothing must not decode anything.
func TestMutateSkipsValidationWhenNothingIsStored(t *testing.T) {
	for _, newH := range allMutateHarnesses(t) {
		h := newH(t)
		t.Run(h.name, func(t *testing.T) {
			for _, act := range []struct {
				name string
				act  RecordMutation
			}{{"unchanged", RecordUnchanged}, {"delete", RecordDelete}} {
				t.Run(act.name, func(t *testing.T) {
					n := countValidates(t, func() {
						if _, _, err := h.mutate(1, "session", func(_ []byte, _ bool) ([]byte, RecordMutation, error) {
							return nil, act.act, nil
						}, CASCond{}); err != nil {
							t.Fatalf("mutate: %v", err)
						}
					})
					if n != 0 {
						t.Fatalf("a %s mutation decoded a record %d times, want 0", act.name, n)
					}
				})
			}
		})
	}
}
