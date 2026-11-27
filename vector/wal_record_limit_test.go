// SPDX-License-Identifier: Apache-2.0

package vector

import (
	"errors"
	"testing"
)

// A record larger than the replay limit (maxWALRecord) cannot be read back: an
// open treats it as damage or, at the end of the log, as a torn tail it cuts.
// Such a write must be refused before it applies, so it is never acknowledged
// and memory never moves ahead of the log.

func lowerWALRecordLimit(t *testing.T, n int) {
	t.Helper()
	prev := walRecordLimit
	walRecordLimit = n
	t.Cleanup(func() { walRecordLimit = prev })
}

func bigTokens(n int) [][]float32 {
	toks := make([][]float32, n)
	for i := range toks {
		toks[i] = []float32{0, 1, 0, 0}
	}
	return toks
}

func TestWALOversizeMultiVectorAddRefusedBeforeApply(t *testing.T) {
	m := seedB2MV(t)
	lowerWALRecordLimit(t, 1024)
	toks := bigTokens(200) // 200 * (4 + 16) bytes, over the lowered limit
	ops := map[string]func() error{
		"AddCAS":                  func() error { _, err := m.AddCAS(9, toks, nil, CASCond{}); return err },
		"AddIfAbsent":             func() error { _, err := m.AddIfAbsent(9, toks, nil); return err },
		"MultiAddIfAbsentVersion": func() error { _, err := m.MultiAddIfAbsentVersion(9, toks, nil, nil, 3); return err },
		"MultiRestoreAdd":         func() error { return m.MultiRestoreAdd(9, toks, nil, nil, 3) },
	}
	for name, op := range ops {
		if err := op(); !errors.Is(err, ErrWALRecordTooLarge) {
			t.Fatalf("%s of an oversize document = %v, want ErrWALRecordTooLarge", name, err)
		}
		if m.Exists(9) {
			t.Fatalf("a refused %s applied in memory", name)
		}
	}
	// Not poisoned: a document that fits is still accepted.
	if _, err := m.AddCAS(9, bigTokens(2), nil, CASCond{}); err != nil {
		t.Fatalf("a document under the limit after a refusal: %v", err)
	}
}

// TestWALOversizeRecordPoisonsLog: the last line, for any family, is the
// append itself. An oversize payload is never written; the log is poisoned so
// no snapshot can make the applied write permanent.
func TestWALOversizeRecordPoisonsLog(t *testing.T) {
	cs := openReplaceStore(t, t.TempDir(), durablePersistentCfg())
	t.Cleanup(func() { _ = cs.Close() })
	c := acquire(t, cs)
	defer c.Release()
	lowerWALRecordLimit(t, 64)
	before := fileSize(t, c.wal.f.Name())
	if err := c.Insert(9, durableVec(9), 0, Metadata{"big": NewString(string(make([]byte, 256)))}, nil); !errors.Is(err, ErrWALRecordTooLarge) {
		t.Fatalf("oversize insert = %v, want ErrWALRecordTooLarge", err)
	}
	if after := fileSize(t, c.wal.f.Name()); after != before {
		t.Fatalf("log grew from %d to %d bytes for a refused record", before, after)
	}
	if !c.wal.poisoned.Load() {
		t.Fatal("log not poisoned after refusing a record already applied in memory")
	}
}
