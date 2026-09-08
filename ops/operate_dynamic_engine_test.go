// SPDX-License-Identifier: Apache-2.0

package ops

// Tests for the dynamic-mode byte engine (design doc §2.9). As with the
// schema engine, the tree oracle in operate_oracle_test.go is the definition
// of correctness: the semantics suite runs against both, and the property
// test below compares them call after call, down to the stored bytes.

import (
	"bytes"
	"errors"
	"fmt"
	"math"
	"math/rand"
	"reflect"
	"testing"

	"github.com/rostamlabs/rostam/sdk/wire"
)

// TestDynamicEngineSemantics runs the whole semantics suite against the byte
// engines with dynamic mode enabled: every "dynamic:" sub-test the schema
// engine skips must hold here.
func TestDynamicEngineSemantics(t *testing.T) {
	runSemantics(t, bytesApply)
}

// TestDynamicEngineMatchesOracle is the equivalence property for dynamic
// mode: for random op lists drawn from the whole vocabulary — including
// CONFIG/TRIM, the control ops, deliberately invalid paths and the occasional
// MIGRATE freeze — the byte engine and the tree oracle must agree on whether
// the call errors, on the result frame, and on the stored bytes, call after
// call, so a divergence that only shows up on an already-diverged record is
// caught too.
func TestDynamicEngineMatchesOracle(t *testing.T) {
	const iters = 2000
	rng := rand.New(rand.NewSource(11)) //nolint:gosec // deterministic test input, not cryptography
	for iter := 0; iter < iters; iter++ {
		seed := rng.Int63()
		sub := rand.New(rand.NewSource(seed)) //nolint:gosec // deterministic test input
		var oracle, byteRec *wire.Record
		for call := 0; call < 5; call++ {
			a := randomDynamicArgs(sub)
			o, ores, oerr := applyTree(oracle, a, int64(iter))
			e, eres, eerr := bytesApply(byteRec, a, int64(iter))
			if (oerr == nil) != (eerr == nil) {
				t.Fatalf("seed %d call %d: oracle err %v, engine err %v\nargs %+v",
					seed, call, oerr, eerr, a)
			}
			if oerr != nil {
				continue
			}
			if !reflect.DeepEqual(ores, eres) {
				t.Fatalf("seed %d call %d: results differ\noracle %+v\nengine %+v\nargs %+v",
					seed, call, ores, eres, a)
			}
			if (o == nil) != (e == nil) {
				t.Fatalf("seed %d call %d: oracle nil=%v engine nil=%v\nargs %+v",
					seed, call, o == nil, e == nil, a)
			}
			if o != nil && !bytes.Equal(o.Encode(), e.Encode()) {
				t.Fatalf("seed %d call %d: bytes differ\noracle %x\nengine %x\nargs %+v",
					seed, call, o.Encode(), e.Encode(), a)
			}
			oracle, byteRec = o, e
		}
	}
}

// --- cross-mode equivalence -------------------------------------------------

// crossOp is one logical op in both spellings: addressed by position against
// a schema record, and by name (with the type the schema declares) against a
// dynamic one.
type crossOp struct {
	schema wire.OperateOp
	dyn    wire.OperateOp
}

// crossModeOps builds a batch of logically identical ops for both modes. It
// draws only ops whose meaning does not depend on which mode stores the
// record: writes to a scalar field or a table column, and DEL of a row or a
// column. (DEL of a FIELD does differ by design — schema mode zeroes it,
// dynamic mode removes it and, with the last one, the record — and the
// semantics suite covers that rule directly.)
func crossModeOps(rng *rand.Rand, s *wire.Schema) []crossOp {
	pool := []uint8{wire.OperateOpSET, wire.OperateOpADD, wire.OperateOpMUL, wire.OperateOpMIN,
		wire.OperateOpMAX, wire.OperateOpAND, wire.OperateOpOR, wire.OperateOpXOR,
		wire.OperateOpSHL, wire.OperateOpSHR, wire.OperateOpSTAMP}
	ops := make([]crossOp, 0, 6)
	for i := 0; i < 1+rng.Intn(6); i++ {
		opcode := pool[rng.Intn(len(pool))]
		a := int64(rng.Intn(300)) - 20
		if opcode == wire.OperateOpSHL || opcode == wire.OperateOpSHR {
			a = int64(rng.Intn(65)) // a shift operand outside [0,64] is an error in both modes
		}
		key := keyU64(uint64(rng.Intn(8)))
		aux := uint8(0)
		if opcode == wire.OperateOpSTAMP {
			aux = uint8(rng.Intn(2))
		}
		switch pick := rng.Intn(10); {
		case pick < 3: // a scalar record field
			pos := rng.Intn(3)
			fd := &s.Fields[pos]
			ops = append(ops, crossOp{
				schema: wire.OperateOp{Opcode: opcode, Type: opFromSchema, Aux: aux, Path: fieldPath(uint32(pos)), A: a},
				dyn:    wire.OperateOp{Opcode: opcode, Type: fd.Type, Aux: aux, Path: namePath(fd.Name), A: a},
			})
		case pick < 8: // a table column
			col := rng.Intn(len(s.Fields[3].Table.Cols))
			cd := &s.Fields[3].Table.Cols[col]
			ops = append(ops, crossOp{
				schema: wire.OperateOp{Opcode: opcode, Type: opFromSchema, Aux: aux,
					Path: colPath(3, key, uint32(col)), A: a},
				dyn: wire.OperateOp{Opcode: opcode, Type: cd.Type, Aux: aux,
					Path: nameColPath(s.Fields[3].Name, key, cd.Name), A: a},
			})
		case pick == 8: // DEL of one column
			col := rng.Intn(len(s.Fields[3].Table.Cols))
			cd := &s.Fields[3].Table.Cols[col]
			ops = append(ops, crossOp{
				schema: wire.OperateOp{Opcode: wire.OperateOpDEL, Type: opFromSchema, Path: colPath(3, key, uint32(col))},
				dyn: wire.OperateOp{Opcode: wire.OperateOpDEL, Type: opFromSchema,
					Path: nameColPath(s.Fields[3].Name, key, cd.Name)},
			})
		default: // DEL of one row
			ops = append(ops, crossOp{
				schema: wire.OperateOp{Opcode: wire.OperateOpDEL, Type: opFromSchema, Path: rowPath(3, key)},
				dyn: wire.OperateOp{Opcode: wire.OperateOpDEL, Type: opFromSchema,
					Path: nameRowPath(s.Fields[3].Name, key)},
			})
		}
	}
	return ops
}

// TestCrossModeEquivalence is design doc §5.13's cross-mode property: the
// same logical op list, applied by position to a schema record and by name to
// a dynamic one, must leave the same logical values behind — the same field
// values, the same rows under the same keys, and the same column values,
// even though the two records store none of it the same way.
func TestCrossModeEquivalence(t *testing.T) {
	s := sessionSchema()
	rng := rand.New(rand.NewSource(7)) //nolint:gosec // deterministic test input
	var schemaRec, dynRec *wire.Record
	for call := 0; call < 50; call++ {
		batch := crossModeOps(rng, s)
		sa := &wire.OperateArgs{Create: wire.OperateCreateSchema, Schema: s.Encode()}
		da := &wire.OperateArgs{Create: wire.OperateCreateDynamic}
		for _, c := range batch {
			sa.Ops = append(sa.Ops, c.schema)
			da.Ops = append(da.Ops, c.dyn)
		}
		sr, _, serr := bytesApply(schemaRec, sa, int64(call)*1000)
		if serr != nil {
			t.Fatalf("call %d schema mode: %v\nops %+v", call, serr, sa.Ops)
		}
		dr, _, derr := bytesApply(dynRec, da, int64(call)*1000)
		if derr != nil {
			t.Fatalf("call %d dynamic mode: %v\nops %+v", call, derr, da.Ops)
		}
		schemaRec, dynRec = sr, dr
		if schemaRec == nil || dynRec == nil {
			t.Fatalf("call %d: a record disappeared (schema nil=%v dynamic nil=%v)", call, sr == nil, dr == nil)
		}
	}

	for i := range s.Fields {
		fd := &s.Fields[i]
		if fd.Type != wire.OperateTypeTable {
			want := schemaRec.Fields[i].Cell
			got := dynCellByName(dynRec.Fields, fd.Name, fd.Type, fd.N)
			if !cellEqual(want, got) {
				t.Fatalf("field %q: schema %+v, dynamic %+v", fd.Name, want, got)
			}
			continue
		}
		stbl := schemaRec.Fields[i].Table
		dfi := treeFindField(dynRec.Fields, fd.Name)
		if dfi < 0 || dynRec.Fields[dfi].Table == nil {
			t.Fatalf("table %q missing from the dynamic record: %+v", fd.Name, dynRec.Fields)
		}
		dtbl := dynRec.Fields[dfi].Table
		if len(stbl.Rows) != len(dtbl.Rows) {
			t.Fatalf("table %q: %d schema rows, %d dynamic rows", fd.Name, len(stbl.Rows), len(dtbl.Rows))
		}
		for r := range stbl.Rows {
			row := &stbl.Rows[r]
			dri, found := treeFindRow(dtbl.Rows, row.Key, 0, true)
			if !found {
				t.Fatalf("table %q: row %x missing from the dynamic record", fd.Name, row.Key)
			}
			for c := range fd.Table.Cols {
				cd := &fd.Table.Cols[c]
				want := row.Cols[c].Cell
				got := dynColByName(dtbl.Rows[dri].Cols, cd.Name, cd.Type, cd.N)
				if !cellEqual(want, got) {
					t.Fatalf("table %q row %x column %q: schema %+v, dynamic %+v",
						fd.Name, row.Key, cd.Name, want, got)
				}
			}
		}
	}
}

// dynCellByName is the dynamic record's value for a schema field, or that
// field's zero when the dynamic record never had it written (design doc §2.4,
// "absent is zero" — which is exactly what an untouched schema field holds).
func dynCellByName(fields []wire.Field, name string, typ, n uint8) wire.Cell {
	if i := treeFindField(fields, name); i >= 0 {
		return fields[i].Cell
	}
	return wire.ZeroCell(typ, n)
}

func dynColByName(cols []wire.Col, name string, typ, n uint8) wire.Cell {
	if i := treeFindCol(cols, name); i >= 0 {
		return cols[i].Cell
	}
	return wire.ZeroCell(typ, n)
}

// cellEqual compares two cells by value: floats by bit pattern (so a
// canonical NaN equals itself) and bytes by content, since an empty payload
// is spelled both nil and []byte{}.
func cellEqual(a, b wire.Cell) bool {
	return a.Type == b.Type && a.N == b.N && a.U == b.U &&
		math.Float64bits(a.F) == math.Float64bits(b.F) && bytes.Equal(a.B, b.B)
}

// --- allocations ------------------------------------------------------------

// bigDynamicRecord builds the design doc's §4 session shape as a dynamic
// record with n rows in its table: the same worked example the schema
// engine's allocation budget is quoted against, self-describing instead.
func bigDynamicRecord(t *testing.T, n int) []byte {
	t.Helper()
	a := &wire.OperateArgs{Create: wire.OperateCreateDynamic, Ops: []wire.OperateOp{
		{Opcode: wire.OperateOpSET, Type: wire.OperateTypeU8, Path: namePath("rc"), A: 1},
		{Opcode: wire.OperateOpSET, Type: wire.OperateTypeU8, Path: namePath("bc"), A: 1},
		{Opcode: wire.OperateOpSET, Type: wire.OperateTypeU32, Path: namePath("hist"), A: 1},
	}}
	for k := 0; k < n; k++ {
		a.Ops = append(a.Ops, wire.OperateOp{Opcode: wire.OperateOpADD, Type: wire.OperateTypeU16,
			Path: nameColPath("b", keyU64(uint64(k)), "c"), A: 1})
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

// dynamicArgsFor is the worked example's per-request op list against one
// existing row: six ops that all patch fixed-width cells in place.
func dynamicArgsFor(key []byte) *wire.OperateArgs {
	return &wire.OperateArgs{Create: wire.OperateCreateDynamic, Ops: []wire.OperateOp{
		{Opcode: wire.OperateOpADD, Type: wire.OperateTypeU8, Path: namePath("rc"), A: 1},
		{Opcode: wire.OperateOpADD, Type: wire.OperateTypeU8, Path: namePath("bc"), A: 1},
		{Opcode: wire.OperateOpOR, Type: wire.OperateTypeU32, Path: namePath("hist"), A: 4},
		{Opcode: wire.OperateOpADD, Type: wire.OperateTypeU16, Path: nameColPath("b", key, "c"), A: 1},
		{Opcode: wire.OperateOpMAX, Type: wire.OperateTypeI64, Path: nameColPath("b", key, "hi"), A: 500},
		{Opcode: wire.OperateOpSTAMP, Type: wire.OperateTypeU32, Aux: wire.OperateStampS,
			Path: nameColPath("b", key, "t")},
	}}
}

// TestDynamicEngineAllocs pins the §2.6 budget for dynamic mode's in-place
// path: the record copy and the result frame, and nothing per op. A
// regression here means a lookup or a patch started building something —
// exactly the cost the byte engine exists to avoid.
func TestDynamicEngineAllocs(t *testing.T) {
	rec := bigDynamicRecord(t, 1024)
	a := dynamicArgsFor(keyU64(512))
	a.Rets = nil
	allocs := testing.AllocsPerRun(50, func() {
		if _, _, _, err := applyRecordBytes(rec, a, 1); err != nil {
			t.Fatal(err)
		}
	})
	if allocs > 3 {
		t.Fatalf("%v allocations per apply, want at most 3", allocs)
	}
	t.Logf("allocations per apply: %v", allocs)
}

// --- freeze -----------------------------------------------------------------

// TestDynamicFreezeThenSchemaCallsWork covers the handover a freeze makes
// (design doc §2.9): once MIGRATE has rewritten the record in schema mode,
// the rest of the same call and every later call must see a schema record —
// including the create parameter, which now has to say SCHEMA.
func TestDynamicFreezeThenSchemaCallsWork(t *testing.T) {
	base, _, _, err := applyRecordBytes(nil, &wire.OperateArgs{Create: wire.OperateCreateDynamic,
		Ops: []wire.OperateOp{
			{Opcode: wire.OperateOpSET, Type: wire.OperateTypeU32, Path: namePath("hits"), A: 7},
			{Opcode: wire.OperateOpSET, Type: wire.OperateTypeBytes, Path: namePath("name"), Bytes: []byte("ab")},
			{Opcode: wire.OperateOpSET, Type: wire.OperateTypeU16, Path: nameColPath("b", keyU64(9), "c"), A: 3},
		}}, 0)
	if err != nil {
		t.Fatal(err)
	}

	fs := frozenSchema()
	// The ops after the MIGRATE run against the frozen record: by position,
	// which a dynamic record rejects outright.
	frozen, _, res, err := applyRecordBytes(base, &wire.OperateArgs{Create: wire.OperateCreateNone,
		Ops: []wire.OperateOp{
			{Opcode: wire.OperateOpMIGRATE, Type: opFromSchema, Path: recPath(),
				A: wire.OperateMigrateFromDynamic, Bytes: fs.Encode()},
			{Opcode: wire.OperateOpADD, Type: opFromSchema, Path: fieldPath(0), A: 5},
		},
		Rets: []wire.OperateRet{
			{Mode: wire.OperateRetValue, Path: fieldPath(0)},
			{Mode: wire.OperateRetCount, Path: fieldPath(2)},
		}}, 0)
	if err != nil {
		t.Fatalf("freeze: %v", err)
	}
	if len(frozen) == 0 || frozen[0] != wire.OperateModeSchema {
		t.Fatalf("record is not in schema mode: %x", frozen)
	}
	if c, _, derr := wire.DecodeTaggedCell(res.Values[0]); derr != nil || c.U != 12 {
		t.Fatalf("the op after the MIGRATE did not see the frozen record: %+v %v", c, derr)
	}
	if c, _, derr := wire.DecodeTaggedCell(res.Values[1]); derr != nil || c.U != 1 {
		t.Fatalf("row COUNT after the freeze: %+v %v", c, derr)
	}

	// A later call must speak schema mode, at the frozen version.
	after, _, _, err := applyRecordBytes(frozen, &wire.OperateArgs{Create: wire.OperateCreateSchema,
		Schema: fs.Encode(),
		Ops:    []wire.OperateOp{{Opcode: wire.OperateOpADD, Type: opFromSchema, Path: fieldPath(0), A: 1}}}, 0)
	if err != nil {
		t.Fatalf("schema-mode call on a frozen record: %v", err)
	}
	rec, err := wire.DecodeRecord(after)
	if err != nil {
		t.Fatal(err)
	}
	if rec.Fields[0].Cell.U != 13 || string(rec.Fields[1].Cell.B) != "ab" {
		t.Fatalf("frozen values not carried: %+v", rec.Fields)
	}
	if _, _, _, err := applyRecordBytes(frozen, &wire.OperateArgs{Create: wire.OperateCreateDynamic,
		Ops: []wire.OperateOp{{Opcode: wire.OperateOpADD, Type: wire.OperateTypeU8, Path: namePath("hits"), A: 1}}},
		0); !errors.Is(err, wire.ErrOperateMode) {
		t.Fatalf("a dynamic call on a frozen record: err = %v, want ErrOperateMode", err)
	}
}

// TestDynamicMigrateRejectsSchemaEvolution pins the other half of §2.9's
// migrate rule: only a freeze can migrate a dynamic record, since there is no
// stored schema for an append-only extension to extend.
func TestDynamicMigrateRejectsSchemaEvolution(t *testing.T) {
	base, _, _, err := applyRecordBytes(nil, &wire.OperateArgs{Create: wire.OperateCreateDynamic,
		Ops: []wire.OperateOp{{Opcode: wire.OperateOpSET, Type: wire.OperateTypeU8, Path: namePath("a"), A: 1}}}, 0)
	if err != nil {
		t.Fatal(err)
	}
	fs := frozenSchema()
	if _, _, _, err := applyRecordBytes(base, &wire.OperateArgs{Create: wire.OperateCreateNone,
		Ops: []wire.OperateOp{{Opcode: wire.OperateOpMIGRATE, Type: opFromSchema, Path: recPath(),
			A: 3, Bytes: fs.Encode()}}}, 0); !errors.Is(err, wire.ErrOperateMode) {
		t.Fatalf("err = %v, want ErrOperateMode", err)
	}
	// A malformed blob is reported as such whatever the from-version says.
	if _, _, _, err := applyRecordBytes(base, &wire.OperateArgs{Create: wire.OperateCreateNone,
		Ops: []wire.OperateOp{{Opcode: wire.OperateOpMIGRATE, Type: opFromSchema, Path: recPath(),
			A: wire.OperateMigrateFromDynamic, Bytes: []byte{9, 9}}}}, 0); !errors.Is(err, wire.ErrShortArgs) {
		t.Fatalf("err = %v, want ErrShortArgs", err)
	}
}

// --- hostile input ----------------------------------------------------------

// TestDynamicEngineNeverPanics feeds applyRecordBytes dynamic-mode bytes it
// must treat as hostile: a plain `put` can store anything under an operate
// key, so every offset in the engine comes from a bounds-checked walk.
// Truncated records, single-byte mutations and the shared hostile frames must
// all come back as an error with no output, never a panic — and a record the
// codec accepts must still be one after the call.
func TestDynamicEngineNeverPanics(t *testing.T) {
	valid := bigDynamicRecord(t, 4)
	a := dynamicArgsFor(keyU64(2))
	a.Rets = []wire.OperateRet{
		{Mode: wire.OperateRetValue, Path: recPath()},
		{Mode: wire.OperateRetValue, Path: namePath("b")},
		{Mode: wire.OperateRetValue, Path: nameRowPath("b", keyU64(2))},
		{Mode: wire.OperateRetCount, Path: namePath("b")},
	}
	del := &wire.OperateArgs{Create: wire.OperateCreateNone, Ops: []wire.OperateOp{
		{Opcode: wire.OperateOpDEL, Type: opFromSchema, Path: nameRowPath("b", keyU64(1))},
		{Opcode: wire.OperateOpDEL, Type: opFromSchema, Path: nameColPath("b", keyU64(3), "c")},
		{Opcode: wire.OperateOpTRIM, Type: opFromSchema, Path: namePath("b"), A: 1,
			Aux: wire.OperatePolicyMinCol, Bytes: []byte("c")},
	}}
	cfg := &wire.OperateArgs{Create: wire.OperateCreateDynamic, Ops: []wire.OperateOp{
		{Opcode: wire.OperateOpCONFIG, Type: opFromSchema, Path: namePath("b"), A: 2,
			Aux: wire.OperatePolicyMaxCol, Bytes: []byte("hi")},
		{Opcode: wire.OperateOpSET, Type: wire.OperateTypeBytes, Path: namePath("zz"), Bytes: []byte("xy")},
		{Opcode: wire.OperateOpMIGRATE, Type: opFromSchema, Path: recPath(),
			A: wire.OperateMigrateFromDynamic, Bytes: frozenSchema().Encode()},
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
	for _, body := range hostileBodies() { // the shared hostile frames, as dynamic records
		corpus = append(corpus, append([]byte{wire.OperateModeDynamic}, body...))
	}

	for _, args := range []*wire.OperateArgs{a, del, cfg} {
		for _, in := range corpus {
			before := append([]byte(nil), in...)
			out, deleted, res, err := safeApply(in, args)
			if err != nil {
				if out != nil || deleted || res != nil {
					t.Fatalf("error path returned output for %x: out=%x deleted=%v res=%+v", in, out, deleted, res)
				}
			} else if out != nil {
				if _, derr := wire.DecodeRecord(in); derr != nil {
					// The engine's guarantee is valid in, valid out: bytes the
					// codec itself rejects are not held to it.
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

// TestNewDynamicEngineValidates pins the structural walk: a frame that does
// not describe itself exactly is a malformed record, not something to patch.
func TestNewDynamicEngineValidates(t *testing.T) {
	valid := bigDynamicRecord(t, 2)
	if _, err := newDynamicEngine(append([]byte(nil), valid...)); err != nil {
		t.Fatalf("a valid record must open: %v", err)
	}
	bad := map[string][]byte{
		"empty":            {},
		"schema mode byte": {wire.OperateModeSchema, 0},
		"unknown mode":     {7, 0},
		"no field count":   {wire.OperateModeDynamic},
		"count overshoots": {wire.OperateModeDynamic, 4},
		"trailing garbage": append(append([]byte(nil), valid...), 0),
		"non-canonical count": append([]byte{wire.OperateModeDynamic, 0x80, 0x00},
			valid[2:]...),
		"names out of order": {wire.OperateModeDynamic, 2, 1, 'b', wire.OperateTypeU8, 0, 1, 'a', wire.OperateTypeU8, 0},
		"duplicate names":    {wire.OperateModeDynamic, 2, 1, 'a', wire.OperateTypeU8, 0, 1, 'a', wire.OperateTypeU8, 0},
		"table type as a cell": {wire.OperateModeDynamic, 1, 1, 'a', wire.OperateTypeTable, 3,
			0, 0, 0},
		"zero-length row key": {wire.OperateModeDynamic, 1, 1, 'a', wire.OperateTypeTable, 6,
			0, 0, 0, 1, 2, 0, 0},
	}
	for name, in := range bad {
		if _, err := newDynamicEngine(append([]byte(nil), in...)); !errors.Is(err, wire.ErrOperateRecord) {
			t.Errorf("%s: err=%v, want ErrOperateRecord", name, err)
		}
	}
}

// TestDynamicRecordSizeCap covers §2.7's record-size backstop in dynamic
// mode: growth is charged before the splice allocates for it, so a call that
// would store an over-large record fails with the record unchanged.
func TestDynamicRecordSizeCap(t *testing.T) {
	restore := maxOperateRecordBytes
	maxOperateRecordBytes = 512
	t.Cleanup(func() { maxOperateRecordBytes = restore })

	small, _, _, err := applyRecordBytes(nil, &wire.OperateArgs{Create: wire.OperateCreateDynamic,
		Ops: []wire.OperateOp{{Opcode: wire.OperateOpSET, Type: wire.OperateTypeBytes,
			Path: namePath("b"), Bytes: bytes.Repeat([]byte{7}, 64)}}}, 0)
	if err != nil {
		t.Fatalf("a record under the bound must still apply: %v", err)
	}
	before := append([]byte(nil), small...)
	out, _, _, err := applyRecordBytes(small, &wire.OperateArgs{Create: wire.OperateCreateDynamic,
		Ops: []wire.OperateOp{{Opcode: wire.OperateOpSET, Type: wire.OperateTypeBytes,
			Path: namePath("b"), Bytes: bytes.Repeat([]byte{7}, 1024)}}}, 0)
	if !errors.Is(err, wire.ErrOperateCap) {
		t.Fatalf("err=%v, want ErrOperateCap", err)
	}
	if out != nil {
		t.Fatalf("a cap hit stored %x", out)
	}
	if !bytes.Equal(small, before) {
		t.Fatal("record changed by a call that hit the record-size cap")
	}
}

// TestDynamicEngineIndexGrows covers the field index's fallback off the
// inline array: a record with more fields than inlineDynFields must index,
// patch and re-index exactly as a small one does.
func TestDynamicEngineIndexGrows(t *testing.T) {
	a := &wire.OperateArgs{Create: wire.OperateCreateDynamic}
	const n = inlineDynFields * 3
	for i := 0; i < n; i++ {
		a.Ops = append(a.Ops, wire.OperateOp{Opcode: wire.OperateOpSET, Type: wire.OperateTypeU32,
			Path: namePath(fmt.Sprintf("f%02d", i)), A: int64(i)})
	}
	out, _, _, err := applyRecordBytes(nil, a, 0)
	if err != nil {
		t.Fatal(err)
	}
	rec, err := wire.DecodeRecord(out)
	if err != nil {
		t.Fatal(err)
	}
	if len(rec.Fields) != n {
		t.Fatalf("%d fields, want %d", len(rec.Fields), n)
	}
	for i := range rec.Fields {
		if rec.Fields[i].Name != fmt.Sprintf("f%02d", i) || rec.Fields[i].Cell.U != uint64(i) {
			t.Fatalf("field %d: %+v", i, rec.Fields[i])
		}
	}
}

// TestDynamicEngineAcceptsExactlyTheCodec pins the engine's structural walk
// against wire.DecodeRecord: the two must agree, byte stream for byte stream,
// on which dynamic records exist at all. If the engine were the more
// permissive of the two it could patch a frame the codec cannot read back;
// if it were the stricter one it would reject records a follower stored.
func TestDynamicEngineAcceptsExactlyTheCodec(t *testing.T) {
	valid := bigDynamicRecord(t, 3)
	var corpus [][]byte
	for i := 0; i <= len(valid); i++ {
		corpus = append(corpus, append([]byte(nil), valid[:i]...))
	}
	for i := range valid {
		for _, delta := range []byte{1, 2, 0x7F, 0x80, 0xFF} {
			m := append([]byte(nil), valid...)
			m[i] += delta
			corpus = append(corpus, m)
		}
	}
	for _, body := range hostileBodies() {
		corpus = append(corpus, append([]byte{wire.OperateModeDynamic}, body...))
	}

	agreed := 0
	for _, in := range corpus {
		_, cerr := wire.DecodeRecord(append([]byte(nil), in...))
		_, eerr := newDynamicEngine(append([]byte(nil), in...))
		if (cerr == nil) != (eerr == nil) {
			t.Fatalf("codec err=%v, engine err=%v for %x", cerr, eerr, in)
		}
		if cerr == nil {
			agreed++
		}
	}
	if agreed < 2 {
		t.Fatalf("only %d corpus entries were valid records — the comparison is vacuous", agreed)
	}
	t.Logf("%d corpus entries, %d of them valid records", len(corpus), agreed)
}
