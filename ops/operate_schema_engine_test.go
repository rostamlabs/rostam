// SPDX-License-Identifier: Apache-2.0

package ops

// Tests for the schema-mode byte engine (design doc §2.6). The engine's
// correctness is defined by the tree oracle in operate_oracle_test.go: the
// semantics suite runs against both, and the property test below compares
// them op-for-op on random schemas and random op lists, down to the stored
// bytes.

import (
	"bytes"
	"errors"
	"fmt"
	"math"
	"math/rand"
	"reflect"
	"sync"
	"testing"

	"github.com/rostamlabs/rostam/sdk/wire"
)

// TestSchemaEngineSemantics runs the whole semantics suite against the byte
// engine with dynamic mode switched off: this is the schema-only lane, which
// keeps the skipDynamic mechanism exercised and pins that every schema-mode
// and mode-independent rule holds without the dynamic engine ever running.
// TestDynamicEngineSemantics runs the same suite with both modes enabled.
func TestSchemaEngineSemantics(t *testing.T) {
	skipDynamic = true
	t.Cleanup(func() { skipDynamic = false })
	runSemantics(t, bytesApply)
}

// TestSchemaEngineMatchesOracle is the equivalence property: for random
// schemas and random op lists drawn from the whole vocabulary, the byte
// engine and the tree oracle must agree on whether the call errors, on the
// result frame, and on the stored bytes — call after call, so divergences
// that only show up on an already-diverged record are caught too.
func TestSchemaEngineMatchesOracle(t *testing.T) {
	const iters = 2000
	rng := rand.New(rand.NewSource(1)) //nolint:gosec // deterministic test input, not cryptography
	for iter := 0; iter < iters; iter++ {
		seed := rng.Int63()
		sub := rand.New(rand.NewSource(seed)) //nolint:gosec // deterministic test input
		s := randomSchema(sub)
		var oracle, engine *wire.Record
		for call := 0; call < 5; call++ {
			a := randomArgs(sub, s)
			o, ores, oerr := applyTree(oracle, a, int64(iter))
			e, eres, eerr := bytesApply(engine, a, int64(iter))
			if operateErrName(oerr) != operateErrName(eerr) {
				t.Fatalf("seed %d call %d: oracle err %q, engine err %q\nschema %+v\nargs %+v",
					seed, call, operateErrName(oerr), operateErrName(eerr), s, a)
			}
			if oerr != nil {
				continue
			}
			if !reflect.DeepEqual(ores, eres) {
				t.Fatalf("seed %d call %d: results differ\noracle %+v\nengine %+v\nargs %+v",
					seed, call, ores, eres, a)
			}
			if (o == nil) != (e == nil) {
				t.Fatalf("seed %d call %d: oracle nil=%v engine nil=%v\nargs %+v", seed, call, o == nil, e == nil, a)
			}
			if o != nil && !bytes.Equal(o.Encode(), e.Encode()) {
				t.Fatalf("seed %d call %d: bytes differ\noracle %x\nengine %x\nargs %+v",
					seed, call, o.Encode(), e.Encode(), a)
			}
			oracle, engine = o, e
			// A MIGRATE call moves the record to the schema it carries, so
			// later calls in this iteration must speak the new version.
			if ns := migratedSchema(a); ns != nil {
				s = ns
			}
		}
	}
}

// bigSessionRecord builds the design doc's §4 session record with n bidder
// rows: the shape the §2.6 allocation budget is quoted against.
func bigSessionRecord(t *testing.T, n int) []byte {
	t.Helper()
	s := sessionSchema()
	a := &wire.OperateArgs{Create: wire.OperateCreateSchema, Schema: s.Encode()}
	for k := 0; k < n; k++ {
		a.Ops = append(a.Ops, wire.OperateOp{Opcode: wire.OperateOpADD, Type: opFromSchema,
			Path: colPath(3, keyU64(uint64(k)), 0), A: 1})
	}
	out, _, _, err := applyRecordBytes(nil, a, 1)
	if err != nil {
		t.Fatal(err)
	}
	if _, derr := wire.DecodeRecord(out); derr != nil {
		t.Fatal(derr)
	}
	return out
}

// sessionArgsFor is the worked example's per-request op list against one
// existing bidder row: six ops that all patch fixed-width cells in place.
func sessionArgsFor(key []byte) *wire.OperateArgs {
	s := sessionSchema()
	return &wire.OperateArgs{Create: wire.OperateCreateSchema, Schema: s.Encode(), Ops: []wire.OperateOp{
		{Opcode: wire.OperateOpADD, Type: opFromSchema, Path: fieldPath(0), A: 1},
		{Opcode: wire.OperateOpADD, Type: opFromSchema, Path: fieldPath(1), A: 1},
		{Opcode: wire.OperateOpOR, Type: opFromSchema, Path: fieldPath(2), A: 4},
		{Opcode: wire.OperateOpADD, Type: opFromSchema, Path: colPath(3, key, 0), A: 1},
		{Opcode: wire.OperateOpMAX, Type: opFromSchema, Path: colPath(3, key, 1), A: 500},
		{Opcode: wire.OperateOpSTAMP, Type: opFromSchema, Aux: wire.OperateStampS, Path: colPath(3, key, 2)},
	}}
}

// TestSchemaEngineAllocs pins the design doc §2.6 budget for the in-place
// path: one allocation for the record copy and one for the result frame. A
// regression here means the engine started building something per call —
// exactly the cost the byte engine exists to avoid.
func TestSchemaEngineAllocs(t *testing.T) {
	rec := bigSessionRecord(t, 1024)
	a := sessionArgsFor(keyU64(512))
	a.Rets = nil
	allocs := testing.AllocsPerRun(50, func() {
		if _, _, _, err := applyRecordBytes(rec, a, 1); err != nil {
			t.Fatal(err)
		}
	})
	if allocs > 2 {
		t.Fatalf("%v allocations per apply, want at most 2", allocs)
	}
	t.Logf("allocations per apply: %v", allocs)
}

// TestSchemaRecordSizeCap covers §2.7's record-size backstop: a call that
// would store an over-large record fails with ErrOperateCap and the stored
// bytes untouched.
func TestSchemaRecordSizeCap(t *testing.T) {
	s := floatSchema()
	base, _, _, err := applyRecordBytes(nil, &wire.OperateArgs{Create: wire.OperateCreateSchema, Schema: s.Encode(),
		Ops: []wire.OperateOp{{Opcode: wire.OperateOpSET, Type: opFromSchema, Path: fieldPath(2),
			Bytes: bytes.Repeat([]byte{7}, 64)}}}, 0)
	if err != nil {
		t.Fatal(err)
	}
	before := append([]byte(nil), base...)

	restore := maxOperateRecordBytes
	maxOperateRecordBytes = len(base) + 16
	t.Cleanup(func() { maxOperateRecordBytes = restore })

	out, deleted, res, err := applyRecordBytes(base, &wire.OperateArgs{Create: wire.OperateCreateSchema, Schema: s.Encode(),
		Ops: []wire.OperateOp{{Opcode: wire.OperateOpSET, Type: opFromSchema, Path: fieldPath(2),
			Bytes: bytes.Repeat([]byte{7}, 4096)}}}, 0)
	if !errors.Is(err, wire.ErrOperateCap) {
		t.Fatalf("err=%v, want ErrOperateCap", err)
	}
	if out != nil || deleted || res != nil {
		t.Fatalf("a failed call returned out=%x deleted=%v res=%+v", out, deleted, res)
	}
	if !bytes.Equal(base, before) {
		t.Fatal("the stored record was mutated by a failing call")
	}
}

// TestSchemaEngineNeverPanics feeds applyRecordBytes stored bytes it must
// treat as hostile: a plain `put` can store anything under an operate key, so
// every offset in the engine is derived from a bounds-checked walk. Truncated
// records, single-byte mutations and the shared hostile frames must all come
// back as an error with no output, never a panic.
func TestSchemaEngineNeverPanics(t *testing.T) {
	valid := bigSessionRecord(t, 4)
	a := sessionArgsFor(keyU64(2))
	a.Rets = []wire.OperateRet{
		{Mode: wire.OperateRetValue, Path: recPath()},
		{Mode: wire.OperateRetValue, Path: fieldPath(3)},
		{Mode: wire.OperateRetValue, Path: rowPath(3, keyU64(2))},
		{Mode: wire.OperateRetCount, Path: fieldPath(3)},
	}
	del := &wire.OperateArgs{Create: wire.OperateCreateNone, Ops: []wire.OperateOp{
		{Opcode: wire.OperateOpDEL, Type: opFromSchema, Path: rowPath(3, keyU64(1))},
		{Opcode: wire.OperateOpTRIM, Type: opFromSchema, Path: fieldPath(3), A: 1,
			Aux: wire.OperatePolicyMinCol, B: 2},
	}}

	var corpus [][]byte
	for i := 0; i <= len(valid); i++ { // every prefix, including the empty one
		corpus = append(corpus, append([]byte(nil), valid[:i]...))
	}
	for i := range valid { // every single-byte mutation
		for _, delta := range []byte{1, 0x80, 0xFF} {
			m := append([]byte(nil), valid...)
			m[i] += delta
			corpus = append(corpus, m)
		}
	}
	for _, body := range hostileBodies() { // the shared hostile frames, as schema records
		corpus = append(corpus, append([]byte{wire.OperateModeSchema}, body...))
	}

	for _, args := range []*wire.OperateArgs{a, del} {
		for _, in := range corpus {
			before := append([]byte(nil), in...)
			out, deleted, res, err := safeApply(in, args)
			if err != nil {
				if out != nil || deleted || res != nil {
					t.Fatalf("error path returned output for %x: out=%x deleted=%v res=%+v", in, out, deleted, res)
				}
			} else if out != nil {
				// Ruling: the engine bounds-checks the frame but does not
				// re-verify that stored rows are strictly ascending — that is
				// an O(rows) walk on every call, exactly the cost §2.6 exists
				// to avoid, and getting it wrong only ever garbles a record
				// that was already garbled. So the guarantee is valid in,
				// valid out: a record the codec accepts must still be one
				// after the call.
				if _, derr := wire.DecodeRecord(in); derr != nil {
					continue
				}
				if _, derr := wire.DecodeRecord(out); derr != nil {
					t.Fatalf("accepted valid %x and produced undecodable %x: %v", in, out, derr)
				}
			}
			if !bytes.Equal(in, before) {
				t.Fatalf("input mutated: %x -> %x", before, in)
			}
		}
	}
}

// safeApply re-panics with the frame that caused it: a bare panic from deep
// inside the engine says nothing about which of the thousand corpus entries
// reached it, and that input is the whole finding.
func safeApply(cur []byte, a *wire.OperateArgs) (out []byte, deleted bool, res *wire.OperateResult, err error) {
	defer func() {
		if r := recover(); r != nil {
			panic(fmt.Sprintf("applyRecordBytes panicked on %x: %v", cur, r))
		}
	}()
	return applyRecordBytes(cur, a, 1)
}

// TestSchemaBlobLenAgreesWithDecode checks the cache's fast path against the
// authority: for any well-formed blob, the framing walk must report exactly
// the length DecodeSchema consumes, or a cache hit could be keyed on the
// wrong bytes.
func TestSchemaBlobLenAgreesWithDecode(t *testing.T) {
	rng := rand.New(rand.NewSource(3)) //nolint:gosec // deterministic test input
	schemas := []*wire.Schema{sessionSchema(), frozenSchema(), floatSchema(), widenedSchema()}
	for i := 0; i < 500; i++ {
		schemas = append(schemas, randomSchema(rng))
	}
	for _, s := range schemas {
		blob := s.Encode()
		_, n, err := wire.DecodeSchema(blob)
		if err != nil {
			t.Fatalf("%+v: %v", s, err)
		}
		got, ok := schemaBlobLen(blob)
		if !ok || got != n {
			t.Fatalf("schemaBlobLen=%d,%v want %d for %x", got, ok, n, blob)
		}
		// Trailing bytes must not change the answer: a stored record's blob
		// is a prefix of the record.
		got, ok = schemaBlobLen(append(blob, 1, 2, 3))
		if !ok || got != n {
			t.Fatalf("with a suffix: schemaBlobLen=%d,%v want %d", got, ok, n)
		}
	}
}

// TestNewSchemaEngineValidates covers the constructor directly: it must accept
// a well-formed stored record and index its tail, and reject anything whose
// frame does not walk exactly to the end of the buffer.
func TestNewSchemaEngineValidates(t *testing.T) {
	valid := bigSessionRecord(t, 3)
	cache := newSchemaCache(4)

	e, err := newSchemaEngine(append([]byte(nil), valid...), cache)
	if err != nil {
		t.Fatal(err)
	}
	if got := e.rows(3); got != 3 {
		t.Fatalf("rows=%d, want 3", got)
	}
	if e.empty() {
		t.Fatal("a fresh engine reports the record deleted")
	}
	r, err := e.resolve(colPath(3, keyU64(1), 0), false, wire.OperateOpIF, opFromSchema, 0)
	if err != nil || !r.present {
		t.Fatalf("resolve: %v %+v", err, r)
	}
	c, err := e.get(r)
	if err != nil || c.U != 1 || c.Type != wire.OperateTypeU16 {
		t.Fatalf("get: %v %+v", err, c)
	}

	bad := map[string][]byte{
		"empty":            {},
		"unknown mode":     {9},
		"dynamic mode":     {wire.OperateModeDynamic, 0},
		"schema only":      append([]byte{wire.OperateModeSchema}, sessionSchema().Encode()...),
		"truncated tail":   valid[:len(valid)-1],
		"trailing garbage": append(append([]byte(nil), valid...), 0),
	}
	for name, in := range bad {
		if _, err := newSchemaEngine(append([]byte(nil), in...), cache); !errors.Is(err, wire.ErrOperateRecord) {
			t.Errorf("%s: err=%v, want ErrOperateRecord", name, err)
		}
	}
}

// TestSchemaEngineConcurrent exercises the two pieces of state applies share —
// the schema cache and the engine pool — from several goroutines at once, so
// the race detector has something to look at. Each goroutine drives its own
// record, so the results are deterministic despite the sharing.
func TestSchemaEngineConcurrent(t *testing.T) {
	schemas := []*wire.Schema{sessionSchema(), floatSchema(), frozenSchema()}
	var wg sync.WaitGroup
	for g := 0; g < 8; g++ {
		wg.Add(1)
		go func(g int) {
			defer wg.Done()
			s := schemas[g%len(schemas)]
			var cur []byte
			for i := 0; i < 200; i++ {
				a := &wire.OperateArgs{Create: wire.OperateCreateSchema, Schema: s.Encode(),
					Ops: []wire.OperateOp{{Opcode: wire.OperateOpADD, Type: opFromSchema,
						Path: fieldPath(0), A: 1}},
					Rets: []wire.OperateRet{{Mode: wire.OperateRetValue, Path: fieldPath(0)}}}
				out, _, res, err := applyRecordBytes(cur, a, int64(i))
				if err != nil {
					t.Errorf("goroutine %d call %d: %v", g, i, err)
					return
				}
				if res.Status != wire.OperateStatusOK {
					t.Errorf("goroutine %d call %d: status %d", g, i, res.Status)
					return
				}
				cur = out
			}
			if _, err := wire.DecodeRecord(cur); err != nil {
				t.Errorf("goroutine %d: %v", g, err)
			}
		}(g)
	}
	wg.Wait()
}

// TestTableBytesIs64Bit pins the arithmetic MIGRATE's size check depends on. A
// migration may widen every row, so the projected size is rows x the NEW row
// width — at the caps, 2^20 x 4096 = 2^32, which wraps to a small positive on
// a 32-bit int and would slip past the record cap, leaving the rebuild to
// append unbounded inside FSM apply. The product cannot be built as a real
// record (it is 4 GiB), so the arithmetic is tested directly.
func TestTableBytesIs64Bit(t *testing.T) {
	const rows, width = 1 << 20, 4096 // wire.OperateMaxRows x wire.OperateMaxRowWidth
	got := tableBytes(rows, width)
	want := uint64(uvarintLen(rows)) + uint64(rows)*uint64(width)
	if got != want {
		t.Fatalf("tableBytes=%d, want %d", got, want)
	}
	if got <= math.MaxInt32 {
		t.Fatalf("tableBytes=%d does not exceed MaxInt32, so this test proves nothing", got)
	}
	// The naive 32-bit product this replaces wrapped to a value that passes
	// every cap check.
	if wrapped := int32(uint32(got)); wrapped >= 0 && wrapped < 16<<20 { //nolint:gosec // demonstrating the wrap
		t.Logf("a 32-bit int would have seen %d bytes instead of %d", wrapped, got)
	}
	if got <= uint64(maxOperateRecordBytes) {
		t.Fatalf("tableBytes=%d is under the record cap %d", got, maxOperateRecordBytes)
	}
	if tableBytes(-1, width) != 0 || tableBytes(rows, -1) != 0 {
		t.Fatal("negative inputs must not produce a size")
	}
}

// TestMigrateSizeCap drives the same check through a real migration: the
// rebuilt record has to fit maxRecordBytes, and a migration that would blow it
// fails with the record unchanged.
func TestMigrateSizeCap(t *testing.T) {
	s := sessionSchema()
	s.Fields[3].Table.Cap = 0
	base := bigSessionRecord(t, 64)
	var stored []byte
	stored = append(stored, base...)

	// Widen every row by appending eight FIXED(255) columns: 64 rows x 2040
	// new bytes is far past the lowered cap.
	ns := sessionSchema()
	ns.Version = 2
	for i := 0; i < 8; i++ {
		ns.Fields[3].Table.Cols = append(ns.Fields[3].Table.Cols,
			wire.ColumnDef{Name: fmt.Sprintf("w%d", i), Type: wire.OperateTypeFixed, N: 255})
	}
	if err := ns.Validate(); err != nil {
		t.Fatal(err)
	}
	migrate := &wire.OperateArgs{Create: wire.OperateCreateNone, Ops: []wire.OperateOp{
		{Opcode: wire.OperateOpMIGRATE, Type: opFromSchema, Path: recPath(), A: 1, Bytes: ns.Encode()}}}

	// It fits by default...
	out, _, _, err := applyRecordBytes(stored, migrate, 0)
	if err != nil {
		t.Fatalf("migration under the cap: %v", err)
	}
	if _, derr := wire.DecodeRecord(out); derr != nil {
		t.Fatalf("migrated record does not decode: %v", derr)
	}
	// ...and is refused when it does not.
	restore := maxOperateRecordBytes
	maxOperateRecordBytes = len(stored) + 1024
	t.Cleanup(func() { maxOperateRecordBytes = restore })
	out, _, _, err = applyRecordBytes(stored, migrate, 0)
	if !errors.Is(err, wire.ErrOperateCap) {
		t.Fatalf("err=%v, want ErrOperateCap", err)
	}
	if out != nil {
		t.Fatal("a refused migration returned a record to store")
	}
	if !bytes.Equal(stored, base) {
		t.Fatal("the stored record was mutated by a refused migration")
	}
}

// TestSchemaBlobMustBeExact: a call's schema blob is exactly one schema. A
// trailing byte is rejected whether the record exists or not — the oracle
// rejects it in treeOpen either way, and accepting it on one path only would
// make "the same call" mean two things.
func TestSchemaBlobMustBeExact(t *testing.T) {
	s := sessionSchema()
	exact := s.Encode()
	trailing := append(append([]byte(nil), exact...), 0)

	mk := func(blob []byte) *wire.OperateArgs {
		return &wire.OperateArgs{Create: wire.OperateCreateSchema, Schema: blob,
			Ops: []wire.OperateOp{{Opcode: wire.OperateOpADD, Type: opFromSchema, Path: fieldPath(0), A: 1}}}
	}
	base, _, _, err := applyRecordBytes(nil, mk(exact), 0)
	if err != nil {
		t.Fatal(err)
	}
	for _, tc := range []struct {
		name string
		cur  []byte
	}{{"absent record", nil}, {"existing record", base}} {
		out, _, _, err := applyRecordBytes(tc.cur, mk(trailing), 0)
		if !errors.Is(err, wire.ErrOperateSchema) {
			t.Errorf("%s: err=%v, want ErrOperateSchema", tc.name, err)
		}
		if out != nil {
			t.Errorf("%s: returned a record to store", tc.name)
		}
	}
	// The oracle agrees, on both paths.
	for _, tc := range []struct {
		name string
		rec  *wire.Record
	}{{"absent record", nil}, {"existing record", mustDecode(t, base)}} {
		if _, _, err := applyTree(tc.rec, mk(trailing), 0); !errors.Is(err, wire.ErrOperateSchema) {
			t.Errorf("oracle, %s: err=%v, want ErrOperateSchema", tc.name, err)
		}
	}
}

func mustDecode(t *testing.T, b []byte) *wire.Record {
	t.Helper()
	rec, err := wire.DecodeRecord(b)
	if err != nil {
		t.Fatal(err)
	}
	return rec
}

// TestMigrateSizeCapAllFixedTarget covers the record-size check for a target
// schema with no variable-length fields at all: the per-tail-field check
// inside rebuiltSize's loop never runs there, so the cap has to be tested
// unconditionally as well. A maximal all-fixed schema (OperateMaxFields
// FIXED(255) fields) is ~16.7 MB of fixed area on its own, past the 16 MiB
// record cap, with no table anywhere to trip the in-loop check.
func TestMigrateSizeCapAllFixedTarget(t *testing.T) {
	s := &wire.Schema{Version: 1, Fields: []wire.FieldDef{{Name: "f0", Type: wire.OperateTypeU64}}}
	base, _, _, err := applyRecordBytes(nil, &wire.OperateArgs{Create: wire.OperateCreateSchema,
		Schema: s.Encode(), Ops: []wire.OperateOp{
			{Opcode: wire.OperateOpADD, Type: opFromSchema, Path: fieldPath(0), A: 7}}}, 0)
	if err != nil {
		t.Fatal(err)
	}
	stored := append([]byte(nil), base...)

	ns := &wire.Schema{Version: 2, Fields: append([]wire.FieldDef(nil), s.Fields...)}
	for i := 0; i < 40; i++ { // 40 x 255 = 10200 bytes of fixed area, no tail at all
		ns.Fields = append(ns.Fields, wire.FieldDef{
			Name: fmt.Sprintf("w%d", i), Type: wire.OperateTypeFixed, N: 255})
	}
	if err := ns.Validate(); err != nil {
		t.Fatal(err)
	}
	layout, err := ns.Layout()
	if err != nil {
		t.Fatal(err)
	}
	if len(layout.VarTail) != 0 {
		t.Fatalf("the target schema must have an empty tail for this test to mean anything: %v", layout.VarTail)
	}
	migrate := &wire.OperateArgs{Create: wire.OperateCreateNone, Ops: []wire.OperateOp{
		{Opcode: wire.OperateOpMIGRATE, Type: opFromSchema, Path: recPath(), A: 1, Bytes: ns.Encode()}}}

	// It fits under the default cap.
	out, _, _, err := applyRecordBytes(stored, migrate, 0)
	if err != nil {
		t.Fatalf("migration under the cap: %v", err)
	}
	if _, derr := wire.DecodeRecord(out); derr != nil {
		t.Fatalf("migrated record does not decode: %v", derr)
	}

	// It does not fit under a lowered one. The assertion that matters is on
	// rebuiltSize itself: applyRecordBytes also checks the finished record, so
	// an end-to-end ErrOperateCap alone would pass even with the cap check
	// missing — while rebuild would already have allocated and filled the
	// oversized buffer to get there.
	restore := maxOperateRecordBytes
	maxOperateRecordBytes = 1024
	t.Cleanup(func() { maxOperateRecordBytes = restore })

	e, err := newSchemaEngine(append([]byte(nil), stored...), operateSchemas)
	if err != nil {
		t.Fatal(err)
	}
	ent, err := operateSchemas.lookupCall(ns.Encode())
	if err != nil {
		t.Fatal(err)
	}
	if len(ent.layout.VarTail) != 0 {
		t.Fatal("the cached target entry must have an empty tail")
	}
	if _, err := e.rebuiltSize(ent); !errors.Is(err, wire.ErrOperateCap) {
		t.Fatalf("rebuiltSize err=%v, want ErrOperateCap — an all-fixed target schema reaches "+
			"no per-tail-field check, so the size must be tested unconditionally", err)
	}

	out, deleted, res, err := applyRecordBytes(stored, migrate, 0)
	if !errors.Is(err, wire.ErrOperateCap) {
		t.Fatalf("err=%v, want ErrOperateCap", err)
	}
	if out != nil || deleted || res != nil {
		t.Fatalf("a refused migration returned out=%x deleted=%v res=%+v", out, deleted, res)
	}
	if !bytes.Equal(stored, base) {
		t.Fatal("the stored record was mutated by a refused migration")
	}
}
