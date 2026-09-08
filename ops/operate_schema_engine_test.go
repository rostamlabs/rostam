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
	"math/rand"
	"reflect"
	"sync"
	"testing"

	"github.com/rostamlabs/rostam/sdk/wire"
)

// TestSchemaEngineSemantics runs the whole semantics suite against the byte
// engine. Dynamic-mode sub-tests are skipped until the dynamic engine lands
// (Task 10); every other rule in the suite must hold here.
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
			if (oerr == nil) != (eerr == nil) {
				t.Fatalf("seed %d call %d: oracle err %v, engine err %v\nschema %+v\nargs %+v",
					seed, call, oerr, eerr, s, a)
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

// safeApply turns a panic into a test failure signal rather than tearing the
// run down, so the frame that caused it is reported.
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
	r, err := e.resolve(colPath(3, keyU64(1), 0), false, opFromSchema, 0)
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
