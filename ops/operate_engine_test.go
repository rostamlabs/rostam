// SPDX-License-Identifier: Apache-2.0

package ops

import (
	"errors"
	"fmt"
	"math"
	"testing"

	"github.com/rostamlabs/rostam/sdk/wire"
)

// --- fakeEngine --------------------------------------------------------
//
// fakeEngine is a minimal engine backed by maps keyed by the textual form of
// an OperatePath (pathKey below), used only to exercise applyOps/evalRets'
// dispatch decisions in isolation from any real byte layout. It records
// every call it receives (in order) so a test can assert what did, and did
// not, run.

type fakeEngine struct {
	// cells holds present scalar values (refKindScalar), keyed by pathKey.
	cells map[string]wire.Cell
	// tables holds present table fields (refKindTable) -> row count, keyed
	// by pathKey of the field path.
	tables map[string]uint64
	// rows holds present rows (refKindRow), keyed by pathKey of the row
	// path.
	rows map[string]bool
	// recordPresent is the ref for OperatePathRecord's present (whether the
	// record existed before this call).
	recordPresent bool
	// recordFieldCount is refKindRecord's count().
	recordFieldCount uint64

	// values overrides the tagged bytes value() returns for a pathKey; a
	// missing entry falls back to AppendTaggedCell of the cell/a placeholder.
	values map[string][]byte

	// forceResolveErr, keyed by pathKey, makes resolve fail for that path.
	forceResolveErr map[string]error

	calls []string

	migrateErr error
	migrated   []migrateCall
	configured []configCall
	trimmed    []trimCall
	deleted    []string

	isEmpty bool
}

type migrateCall struct {
	from   int64
	flags  uint8
	schema []byte
}

type configCall struct {
	capN      uint32
	policy    uint8
	byColName string
}

type trimCall struct {
	keep      uint32
	policy    uint8
	byCol     uint16
	byColName string
}

func newFakeEngine() *fakeEngine {
	return &fakeEngine{
		cells:           map[string]wire.Cell{},
		tables:          map[string]uint64{},
		rows:            map[string]bool{},
		values:          map[string][]byte{},
		forceResolveErr: map[string]error{},
	}
}

// pathKey renders an OperatePath as a stable string so the fake can key its
// maps by it. It is test-only plumbing, not a wire format.
func pathKey(p wire.OperatePath) string {
	switch p.Kind {
	case wire.OperatePathRecord:
		return "record"
	case wire.OperatePathField:
		return "field:" + segKey(p.Field)
	case wire.OperatePathRow:
		return "field:" + segKey(p.Field) + "/row:" + string(p.Key)
	case wire.OperatePathCol:
		return "field:" + segKey(p.Field) + "/row:" + string(p.Key) + "/col:" + segKey(p.Col)
	default:
		return fmt.Sprintf("invalid:%d", p.Kind)
	}
}

func segKey(s wire.OperateSeg) string {
	if s.ByName {
		return "name=" + s.Name
	}
	return fmt.Sprintf("pos=%d", s.Pos)
}

func (f *fakeEngine) resolve(p wire.OperatePath, create bool, opcode, typ, n uint8) (ref, error) {
	_ = opcode // the fake never retypes a stored cell, so the opcode is noise here
	key := pathKey(p)
	f.calls = append(f.calls, "resolve:"+key)
	if err, ok := f.forceResolveErr[key]; ok {
		return ref{}, err
	}
	switch p.Kind {
	case wire.OperatePathRecord:
		return ref{kind: refKindRecord, present: f.recordPresent}, nil
	case wire.OperatePathRow:
		present := f.rows[key]
		if !present && create {
			f.rows[key] = true
			present = true
		}
		return ref{kind: refKindRow, off: hashKey(key), present: present}, nil
	default: // Field or Col: a table field, or a scalar
		if _, ok := f.tables[key]; ok {
			return ref{kind: refKindTable, off: hashKey(key), present: true}, nil
		}
		c, present := f.cells[key]
		if !present {
			if create {
				c = wire.ZeroCell(typ, n)
				f.cells[key] = c
				present = true
			} else {
				c = wire.ZeroCell(typ, n)
			}
		}
		return ref{kind: refKindScalar, off: hashKey(key), typ: c.Type, n: c.N, present: present}, nil
	}
}

func hashKey(s string) int {
	h := 0
	for i := 0; i < len(s); i++ {
		h = h*31 + int(s[i])
	}
	return h
}

func (f *fakeEngine) get(r ref) (wire.Cell, error) {
	f.calls = append(f.calls, "get")
	for key, c := range f.cells {
		if hashKey(key) == r.off {
			return c, nil
		}
	}
	return wire.ZeroCell(r.typ, r.n), nil
}

func (f *fakeEngine) set(r ref, c wire.Cell) error {
	f.calls = append(f.calls, "set")
	for key := range f.cells {
		if hashKey(key) == r.off {
			f.cells[key] = c
			return nil
		}
	}
	return errors.New("fakeEngine: set on unknown ref")
}

func (f *fakeEngine) del(r ref) error {
	f.calls = append(f.calls, "del")
	switch r.kind {
	case refKindRecord:
		f.deleted = append(f.deleted, "record")
	default:
		for key := range f.cells {
			if hashKey(key) == r.off {
				delete(f.cells, key)
				f.deleted = append(f.deleted, key)
				return nil
			}
		}
	}
	return nil
}

func (f *fakeEngine) count(r ref) (uint64, error) {
	f.calls = append(f.calls, "count")
	switch r.kind {
	case refKindRecord:
		return f.recordFieldCount, nil
	case refKindTable:
		for key, n := range f.tables {
			if hashKey(key) == r.off {
				return n, nil
			}
		}
		return 0, nil
	default: // refKindRow, refKindScalar: presence as 1/0 (design doc's
		// "a plain scalar's count is only exists/absent" ruling, same as a
		// row).
		if r.present {
			return 1, nil
		}
		return 0, nil
	}
}

func (f *fakeEngine) trim(r ref, keep uint32, policy uint8, byCol uint16, byColName string) error {
	f.calls = append(f.calls, "trim")
	f.trimmed = append(f.trimmed, trimCall{keep: keep, policy: policy, byCol: byCol, byColName: byColName})
	return nil
}

func (f *fakeEngine) config(r ref, capN uint32, policy uint8, byColName string) error {
	f.calls = append(f.calls, "config")
	f.configured = append(f.configured, configCall{capN: capN, policy: policy, byColName: byColName})
	return nil
}

func (f *fakeEngine) migrate(from int64, flags uint8, schemaBlob []byte) error {
	f.calls = append(f.calls, "migrate")
	if f.migrateErr != nil {
		return f.migrateErr
	}
	f.migrated = append(f.migrated, migrateCall{from: from, flags: flags, schema: schemaBlob})
	return nil
}

func (f *fakeEngine) value(r ref) ([]byte, error) {
	f.calls = append(f.calls, "value")
	for key := range f.cells {
		if hashKey(key) == r.off {
			if v, ok := f.values[key]; ok {
				return v, nil
			}
			return wire.AppendTaggedCell(nil, f.cells[key]), nil
		}
	}
	return []byte{wire.OperateTypeUnset}, nil
}

func (f *fakeEngine) bytes() []byte { return nil }
func (f *fakeEngine) empty() bool   { return f.isEmpty }

// --- helpers -------------------------------------------------------------

func fakeFieldPath(name string) wire.OperatePath {
	return wire.OperatePath{Kind: wire.OperatePathField, Field: wire.OperateSeg{ByName: true, Name: name}}
}

func fakeRowPath(field, key string) wire.OperatePath {
	return wire.OperatePath{Kind: wire.OperatePathRow, Field: wire.OperateSeg{ByName: true, Name: field}, Key: []byte(key)}
}

func fakeRecordPath() wire.OperatePath {
	return wire.OperatePath{Kind: wire.OperatePathRecord}
}

func (f *fakeEngine) setScalar(path string, c wire.Cell) {
	f.cells[path] = c
}

// --- tests -----------------------------------------------------------------

// (a) IF false skips exactly B ops; B beyond the end is errOperateIfRange
// even when the condition is true.
func TestApplyOpsIfSkip(t *testing.T) {
	f := newFakeEngine()
	f.setScalar(pathKey(fakeFieldPath("x")), wire.Cell{Type: wire.OperateTypeU32, U: 1})

	// IF x == 0 is FALSE (x is 1) -> skip the next 1 op: the SET (index 1)
	// is skipped, ADD at index 2 runs against the original x.
	ops := []wire.OperateOp{
		{Opcode: wire.OperateOpIF, Type: wire.OperateTypeU32, Aux: wire.OperateCmpEQ, Path: fakeFieldPath("x"), A: 0, B: 1},
		{Opcode: wire.OperateOpSET, Type: wire.OperateTypeU32, Path: fakeFieldPath("x"), A: 999},
		{Opcode: wire.OperateOpADD, Type: wire.OperateTypeU32, Path: fakeFieldPath("x"), A: 5},
	}
	status, failed, err := applyOps(f, ops, 0)
	if err != nil || status != wire.OperateStatusOK || failed != 0 {
		t.Fatalf("applyOps: status=%d failed=%d err=%v", status, failed, err)
	}
	c := f.cells[pathKey(fakeFieldPath("x"))]
	if c.U != 6 {
		t.Fatalf("x = %d, want 6 (1+5; SET at index 1 must have been skipped)", c.U)
	}

	// IF false: condition false, skip count reaches past the end ->
	// errOperateIfRange regardless.
	f2 := newFakeEngine()
	f2.setScalar(pathKey(fakeFieldPath("x")), wire.Cell{Type: wire.OperateTypeU32, U: 1})
	ops2 := []wire.OperateOp{
		{Opcode: wire.OperateOpIF, Type: wire.OperateTypeU32, Aux: wire.OperateCmpEQ, Path: fakeFieldPath("x"), A: 0, B: 5},
	}
	_, _, err = applyOps(f2, ops2, 0)
	if !errors.Is(err, errOperateIfRange) {
		t.Fatalf("applyOps: err=%v, want errOperateIfRange", err)
	}

	// Condition true (so the branch is NOT taken) but B still reaches past
	// the end -> validated eagerly regardless of branch outcome.
	f3 := newFakeEngine()
	f3.setScalar(pathKey(fakeFieldPath("x")), wire.Cell{Type: wire.OperateTypeU32, U: 0})
	ops3 := []wire.OperateOp{
		{Opcode: wire.OperateOpIF, Type: wire.OperateTypeU32, Aux: wire.OperateCmpEQ, Path: fakeFieldPath("x"), A: 0, B: 5},
	}
	_, _, err = applyOps(f3, ops3, 0)
	if !errors.Is(err, errOperateIfRange) {
		t.Fatalf("applyOps (condition true): err=%v, want errOperateIfRange", err)
	}

	// Negative B is also out of range.
	f4 := newFakeEngine()
	f4.setScalar(pathKey(fakeFieldPath("x")), wire.Cell{Type: wire.OperateTypeU32, U: 0})
	ops4 := []wire.OperateOp{
		{Opcode: wire.OperateOpIF, Type: wire.OperateTypeU32, Aux: wire.OperateCmpEQ, Path: fakeFieldPath("x"), A: 0, B: -1},
	}
	_, _, err = applyOps(f4, ops4, 0)
	if !errors.Is(err, errOperateIfRange) {
		t.Fatalf("applyOps (negative B): err=%v, want errOperateIfRange", err)
	}
}

// (b) CHECK false returns (CheckFailed, idx, nil) and no `set` runs after.
func TestApplyOpsCheckFailedStopsBeforeSet(t *testing.T) {
	f := newFakeEngine()
	f.setScalar(pathKey(fakeFieldPath("x")), wire.Cell{Type: wire.OperateTypeU32, U: 7})

	ops := []wire.OperateOp{
		{Opcode: wire.OperateOpCHECK, Type: wire.OperateTypeU32, Aux: wire.OperateCmpEQ, Path: fakeFieldPath("x"), A: 0}, // false: 7 != 0
		{Opcode: wire.OperateOpSET, Type: wire.OperateTypeU32, Path: fakeFieldPath("x"), A: 999},
	}
	status, failed, err := applyOps(f, ops, 0)
	if err != nil {
		t.Fatalf("applyOps: unexpected err %v", err)
	}
	if status != wire.OperateStatusCheckFailed || failed != 0 {
		t.Fatalf("applyOps: status=%d failed=%d, want CheckFailed at 0", status, failed)
	}
	for _, call := range f.calls {
		if call == "set" {
			t.Fatalf("set was called after CHECK failed: calls=%v", f.calls)
		}
	}
	if f.cells[pathKey(fakeFieldPath("x"))].U != 7 {
		t.Fatalf("x mutated despite CHECK failure")
	}
}

// (c) unknown opcode -> wire.ErrOperateOpcode.
func TestApplyOpsUnknownOpcode(t *testing.T) {
	f := newFakeEngine()
	ops := []wire.OperateOp{
		{Opcode: 250, Type: wire.OperateTypeU32, Path: fakeFieldPath("x"), A: 1},
	}
	_, _, err := applyOps(f, ops, 0)
	if !errors.Is(err, wire.ErrOperateOpcode) {
		t.Fatalf("applyOps: err=%v, want wire.ErrOperateOpcode", err)
	}
}

// (d) MIGRATE at index > 0 -> error; MIGRATE at index 0 forwards (A, Aux,
// Bytes) to e.migrate.
func TestApplyOpsMigrate(t *testing.T) {
	f := newFakeEngine()
	ops := []wire.OperateOp{
		{Opcode: wire.OperateOpMIGRATE, A: 3, Aux: wire.OperateMigrateDropExtra, Bytes: []byte("schema-blob"), Path: fakeRecordPath()},
	}
	status, _, err := applyOps(f, ops, 0)
	if err != nil || status != wire.OperateStatusOK {
		t.Fatalf("applyOps: status=%d err=%v", status, err)
	}
	if len(f.migrated) != 1 {
		t.Fatalf("migrate was not called: %v", f.calls)
	}
	got := f.migrated[0]
	if got.from != 3 || got.flags != wire.OperateMigrateDropExtra || string(got.schema) != "schema-blob" {
		t.Fatalf("migrate called with %+v, want {3 %d schema-blob}", got, wire.OperateMigrateDropExtra)
	}

	f2 := newFakeEngine()
	ops2 := []wire.OperateOp{
		{Opcode: wire.OperateOpSET, Type: wire.OperateTypeU32, Path: fakeFieldPath("x"), A: 1},
		{Opcode: wire.OperateOpMIGRATE, A: 3, Path: fakeRecordPath()},
	}
	_, _, err = applyOps(f2, ops2, 0)
	if !errors.Is(err, wire.ErrOperateOpcode) {
		t.Fatalf("applyOps (MIGRATE not at 0): err=%v, want wire.ErrOperateOpcode", err)
	}
	if len(f2.migrated) != 0 {
		t.Fatalf("migrate must not have been called: %v", f2.migrated)
	}

	// MIGRATE at index 0 but not addressing the record -> wire.ErrOperatePath,
	// and e.migrate must never be called (migrate() has no path of its own
	// to reject it with).
	f3 := newFakeEngine()
	ops3 := []wire.OperateOp{
		{Opcode: wire.OperateOpMIGRATE, A: 3, Path: fakeFieldPath("x")},
	}
	_, _, err = applyOps(f3, ops3, 0)
	if !errors.Is(err, wire.ErrOperatePath) {
		t.Fatalf("applyOps (MIGRATE not at record path): err=%v, want wire.ErrOperatePath", err)
	}
	if len(f3.migrated) != 0 {
		t.Fatalf("migrate must not have been called: %v", f3.migrated)
	}
}

// CONFIG/TRIM at a row, col, or record path -> wire.ErrOperatePath, without
// ever calling resolve (config()/trim() have no path of their own to reject
// it with, so the loop must).
func TestApplyOpsConfigTrimPathKind(t *testing.T) {
	cases := []struct {
		name string
		op   wire.OperateOp
	}{
		{"config-row", wire.OperateOp{Opcode: wire.OperateOpCONFIG, Path: fakeRowPath("b", "k1"), A: 1}},
		{"config-record", wire.OperateOp{Opcode: wire.OperateOpCONFIG, Path: fakeRecordPath(), A: 1}},
		{"trim-row", wire.OperateOp{Opcode: wire.OperateOpTRIM, Path: fakeRowPath("b", "k1"), A: 1}},
		{"trim-record", wire.OperateOp{Opcode: wire.OperateOpTRIM, Path: fakeRecordPath(), A: 1}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			f := newFakeEngine()
			_, _, err := applyOps(f, []wire.OperateOp{tc.op}, 0)
			if !errors.Is(err, wire.ErrOperatePath) {
				t.Fatalf("%s: err=%v, want wire.ErrOperatePath", tc.name, err)
			}
			for _, call := range f.calls {
				if len(call) >= len("resolve:") && call[:len("resolve:")] == "resolve:" {
					t.Fatalf("%s: resolve was called before the path-kind check rejected the op: %v", tc.name, f.calls)
				}
			}
			if len(f.configured) != 0 || len(f.trimmed) != 0 {
				t.Fatalf("%s: config/trim must not have been called", tc.name)
			}
		})
	}
}

// (e)/(f) evalRets maps RetCount to a tagged U64 and RetValue to e.value();
// absent -> [Unset] for VALUE and a tagged 0 for COUNT.
func TestEvalRets(t *testing.T) {
	f := newFakeEngine()
	f.setScalar(pathKey(fakeFieldPath("present")), wire.Cell{Type: wire.OperateTypeU32, U: 42})

	rets := []wire.OperateRet{
		{Mode: wire.OperateRetValue, Path: fakeFieldPath("present")},
		{Mode: wire.OperateRetCount, Path: fakeFieldPath("present")},
		{Mode: wire.OperateRetValue, Path: fakeFieldPath("absent")},
		{Mode: wire.OperateRetCount, Path: fakeFieldPath("absent")},
	}
	vals, err := evalRets(f, rets)
	if err != nil {
		t.Fatalf("evalRets: %v", err)
	}
	if len(vals) != 4 {
		t.Fatalf("evalRets: got %d values, want 4", len(vals))
	}
	wantValue := wire.AppendTaggedCell(nil, wire.Cell{Type: wire.OperateTypeU32, U: 42})
	if string(vals[0]) != string(wantValue) {
		t.Fatalf("VALUE(present) = %v, want %v", vals[0], wantValue)
	}
	wantCount := wire.AppendTaggedCell(nil, wire.Cell{Type: wire.OperateTypeU64, U: 1})
	if string(vals[1]) != string(wantCount) {
		t.Fatalf("COUNT(present) = %v, want %v", vals[1], wantCount)
	}
	if len(vals[2]) != 1 || vals[2][0] != wire.OperateTypeUnset {
		t.Fatalf("VALUE(absent) = %v, want [Unset]", vals[2])
	}
	wantZero := wire.AppendTaggedCell(nil, wire.Cell{Type: wire.OperateTypeU64, U: 0})
	if string(vals[3]) != string(wantZero) {
		t.Fatalf("COUNT(absent) = %v, want %v", vals[3], wantZero)
	}

	// Absent path must not call value()/count() at all: exactly one of each
	// ran, for the two *present* rets above, and nothing extra from the two
	// absent ones.
	nValue, nCount := 0, 0
	for _, call := range f.calls {
		switch call {
		case "value":
			nValue++
		case "count":
			nCount++
		}
	}
	if nValue != 1 || nCount != 1 {
		t.Fatalf("value()/count() called %d/%d times, want 1/1 (absent rets must skip them)", nValue, nCount)
	}

	// Unknown ret mode -> wire.ErrOperateArgs.
	_, err = evalRets(f, []wire.OperateRet{{Mode: 250, Path: fakeFieldPath("present")}})
	if !errors.Is(err, wire.ErrOperateArgs) {
		t.Fatalf("evalRets(bad mode): err=%v, want wire.ErrOperateArgs", err)
	}
}

// (f) a scalar op forwards `present` from the ref into ApplyScalar (absent
// -> arithmetic on zero).
func TestApplyOpsScalarPresentForwarded(t *testing.T) {
	f := newFakeEngine()
	// "y" is absent: ADD 10 against a zero-valued, absent cell must land on
	// 10 (present is irrelevant to ADD's arithmetic itself, but resolve
	// must vivify with the right zero and get() must report it).
	ops := []wire.OperateOp{
		{Opcode: wire.OperateOpADD, Type: wire.OperateTypeU32, Path: fakeFieldPath("y"), A: 10},
	}
	status, _, err := applyOps(f, ops, 0)
	if err != nil || status != wire.OperateStatusOK {
		t.Fatalf("applyOps: status=%d err=%v", status, err)
	}
	c, ok := f.cells[pathKey(fakeFieldPath("y"))]
	if !ok || c.U != 10 {
		t.Fatalf("y = %+v (present=%v), want U=10", c, ok)
	}

	// CompareCell also receives r.present: EXISTS on an absent scalar
	// path is false even though get() returns a defined zero cell.
	f2 := newFakeEngine()
	ops2 := []wire.OperateOp{
		{Opcode: wire.OperateOpCHECK, Type: wire.OperateTypeU32, Aux: wire.OperateCmpExists, Path: fakeFieldPath("z")},
	}
	status2, failed2, err := applyOps(f2, ops2, 0)
	if err != nil {
		t.Fatalf("applyOps: %v", err)
	}
	if status2 != wire.OperateStatusCheckFailed || failed2 != 0 {
		t.Fatalf("CHECK(EXISTS) on absent scalar: status=%d failed=%d, want CheckFailed@0", status2, failed2)
	}
}

// (g) CONFIG/TRIM A bounds: negative or over wire.OperateMaxRows ->
// wire.ErrOperateCap, checked before resolve vivifies anything.
func TestApplyOpsConfigTrimCap(t *testing.T) {
	cases := []struct {
		name string
		op   uint8
		a    int64
	}{
		{"config-negative", wire.OperateOpCONFIG, -1},
		{"config-over", wire.OperateOpCONFIG, int64(wire.OperateMaxRows) + 1},
		{"trim-negative", wire.OperateOpTRIM, -1},
		{"trim-over", wire.OperateOpTRIM, int64(wire.OperateMaxRows) + 1},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			f := newFakeEngine()
			ops := []wire.OperateOp{
				{Opcode: tc.op, Path: fakeFieldPath("tbl"), A: tc.a, Bytes: []byte("col")},
			}
			_, _, err := applyOps(f, ops, 0)
			if !errors.Is(err, wire.ErrOperateCap) {
				t.Fatalf("%s: err=%v, want wire.ErrOperateCap", tc.name, err)
			}
			for _, call := range f.calls {
				if call == "resolve:"+pathKey(fakeFieldPath("tbl")) {
					t.Fatalf("%s: resolve was called before the cap check rejected the op", tc.name)
				}
			}
		})
	}

	// In-bounds CONFIG/TRIM forward exactly (A, Aux, Bytes) as documented.
	f := newFakeEngine()
	ops := []wire.OperateOp{
		{Opcode: wire.OperateOpCONFIG, Path: fakeFieldPath("tbl"), A: 100, Aux: wire.OperatePolicyMinCol, Bytes: []byte("hi")},
	}
	status, _, err := applyOps(f, ops, 0)
	if err != nil || status != wire.OperateStatusOK {
		t.Fatalf("CONFIG in-bounds: status=%d err=%v", status, err)
	}
	if len(f.configured) != 1 || f.configured[0].capN != 100 || f.configured[0].policy != wire.OperatePolicyMinCol || f.configured[0].byColName != "hi" {
		t.Fatalf("config called with %+v", f.configured)
	}

	f2 := newFakeEngine()
	ops2 := []wire.OperateOp{
		{Opcode: wire.OperateOpTRIM, Path: fakeFieldPath("tbl"), A: 5, Aux: wire.OperatePolicyMinKey, B: 2, Bytes: []byte("bc")},
	}
	status2, _, err := applyOps(f2, ops2, 0)
	if err != nil || status2 != wire.OperateStatusOK {
		t.Fatalf("TRIM in-bounds: status=%d err=%v", status2, err)
	}
	if len(f2.trimmed) != 1 || f2.trimmed[0].keep != 5 || f2.trimmed[0].policy != wire.OperatePolicyMinKey || f2.trimmed[0].byCol != 2 || f2.trimmed[0].byColName != "bc" {
		t.Fatalf("trim called with %+v", f2.trimmed)
	}
}

// DEL () (record path) is terminal: OK immediately, no further ops run.
func TestApplyOpsDelRecordTerminal(t *testing.T) {
	f := newFakeEngine()
	f.recordPresent = true
	ops := []wire.OperateOp{
		{Opcode: wire.OperateOpDEL, Path: fakeRecordPath()},
		{Opcode: wire.OperateOpSET, Type: wire.OperateTypeU32, Path: fakeFieldPath("x"), A: 999},
	}
	status, _, err := applyOps(f, ops, 0)
	if err != nil || status != wire.OperateStatusOK {
		t.Fatalf("applyOps: status=%d err=%v", status, err)
	}
	if len(f.deleted) != 1 || f.deleted[0] != "record" {
		t.Fatalf("del(record) not recorded: %v", f.deleted)
	}
	if _, ok := f.cells[pathKey(fakeFieldPath("x"))]; ok {
		t.Fatalf("SET ran after a terminal record DEL")
	}
}

// A row ref compares by presence (count 1/0), via CompareCount.
func TestApplyOpsRowComparesByPresence(t *testing.T) {
	f := newFakeEngine()
	// Row absent: CHECK ABSENT should pass (true), so status stays OK.
	ops := []wire.OperateOp{
		{Opcode: wire.OperateOpCHECK, Aux: wire.OperateCmpAbsent, Path: fakeRowPath("b", "k1")},
	}
	status, _, err := applyOps(f, ops, 0)
	if err != nil || status != wire.OperateStatusOK {
		t.Fatalf("CHECK ABSENT on absent row: status=%d err=%v", status, err)
	}

	// Now the same row is present: CHECK ABSENT should fail.
	f2 := newFakeEngine()
	f2.rows[pathKey(fakeRowPath("b", "k1"))] = true
	ops2 := []wire.OperateOp{
		{Opcode: wire.OperateOpCHECK, Aux: wire.OperateCmpAbsent, Path: fakeRowPath("b", "k1")},
	}
	status2, failed2, err := applyOps(f2, ops2, 0)
	if err != nil {
		t.Fatalf("applyOps: %v", err)
	}
	if status2 != wire.OperateStatusCheckFailed || failed2 != 0 {
		t.Fatalf("CHECK ABSENT on present row: status=%d failed=%d, want CheckFailed@0", status2, failed2)
	}
}

// A table ref compares by row count, via CompareCount.
func TestApplyOpsTableComparesByCount(t *testing.T) {
	f := newFakeEngine()
	f.tables[pathKey(fakeFieldPath("b"))] = 3
	ops := []wire.OperateOp{
		{Opcode: wire.OperateOpCHECK, Aux: wire.OperateCmpGE, Path: fakeFieldPath("b"), A: 3},
	}
	status, _, err := applyOps(f, ops, 0)
	if err != nil || status != wire.OperateStatusOK {
		t.Fatalf("CHECK count>=3 on a 3-row table: status=%d err=%v", status, err)
	}

	ops2 := []wire.OperateOp{
		{Opcode: wire.OperateOpCHECK, Aux: wire.OperateCmpGE, Path: fakeFieldPath("b"), A: 4},
	}
	status2, failed2, err := applyOps(f, ops2, 0)
	if err != nil {
		t.Fatalf("applyOps: %v", err)
	}
	if status2 != wire.OperateStatusCheckFailed || failed2 != 0 {
		t.Fatalf("CHECK count>=4 on a 3-row table: status=%d failed=%d, want CheckFailed@0", status2, failed2)
	}
}

// A scalar op whose path resolves to the record, a table, or a row (not a
// scalar) is rejected, even though resolve() itself succeeded.
func TestApplyOpsScalarRejectsNonScalarTarget(t *testing.T) {
	cases := []struct {
		name string
		path wire.OperatePath
	}{
		{"record", fakeRecordPath()},
		{"table", fakeFieldPath("b")},
		{"row", fakeRowPath("b", "k1")},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			f := newFakeEngine()
			f.tables[pathKey(fakeFieldPath("b"))] = 0
			ops := []wire.OperateOp{
				{Opcode: wire.OperateOpSET, Type: wire.OperateTypeU32, Path: tc.path, A: 1},
			}
			_, _, err := applyOps(f, ops, 0)
			if !errors.Is(err, wire.ErrOperatePath) {
				t.Fatalf("%s: err=%v, want wire.ErrOperatePath", tc.name, err)
			}
			for _, call := range f.calls {
				if call == "get" || call == "set" {
					t.Fatalf("%s: get/set must not run against a non-scalar ref: %v", tc.name, f.calls)
				}
			}
		})
	}
}

// fixedN: SET creating a FIXED dynamic target passes n=len(o.Bytes); an
// out-of-range FIXED length is rejected before resolve is called.
func TestFixedN(t *testing.T) {
	n, err := fixedN(&wire.OperateOp{Type: wire.OperateTypeFixed, Bytes: []byte("abc")})
	if err != nil || n != 3 {
		t.Fatalf("fixedN(FIXED,3 bytes) = (%d,%v), want (3,nil)", n, err)
	}
	n, err = fixedN(&wire.OperateOp{Type: wire.OperateTypeU32, Bytes: []byte("abc")})
	if err != nil || n != 0 {
		t.Fatalf("fixedN(U32) = (%d,%v), want (0,nil)", n, err)
	}
	_, err = fixedN(&wire.OperateOp{Type: wire.OperateTypeFixed, Bytes: nil})
	if !errors.Is(err, wire.ErrOperateType) {
		t.Fatalf("fixedN(FIXED, empty) = %v, want wire.ErrOperateType", err)
	}
	big := make([]byte, wire.OperateMaxKeyLen+1)
	_, err = fixedN(&wire.OperateOp{Type: wire.OperateTypeFixed, Bytes: big})
	if !errors.Is(err, wire.ErrOperateType) {
		t.Fatalf("fixedN(FIXED, too long) = %v, want wire.ErrOperateType", err)
	}
}

// bytesApply adapts applyRecordBytes to the semantics suite's applier type:
// it encodes the tree the suite hands it, applies the call to those bytes,
// and decodes the result back into a tree. Every byte-level engine is tested
// through this adapter, so a divergence from the oracle shows up as either a
// different tree or bytes the record codec refuses.
func bytesApply(rec *wire.Record, a *wire.OperateArgs, stampMs int64) (*wire.Record, *wire.OperateResult, error) {
	var cur []byte
	if rec != nil {
		cur = rec.Encode()
	}
	out, deleted, res, err := applyRecordBytes(cur, a, stampMs)
	if err != nil {
		return nil, nil, err
	}
	if res != nil && res.Status == wire.OperateStatusCheckFailed {
		// A no-op: nothing is stored, and the record the suite sees is the
		// one it passed in.
		return rec, res, nil
	}
	if deleted || len(out) == 0 {
		return nil, res, nil
	}
	got, derr := wire.DecodeRecord(out)
	if derr != nil {
		return nil, nil, fmt.Errorf("engine produced undecodable bytes %x: %w", out, derr)
	}
	return got, res, nil
}

// TestApplyOpsTrimByCol pins the narrowing of TRIM's eviction-column operand.
// It travels as an int64 and the engines address columns as a uint16, so a
// plain conversion would let 1<<16 alias column 0 and -65536 alias it too —
// silently trimming by a real column where the tree oracle rejects the op.
// Anything outside [0, OperateMaxCols) must arrive as OperateMaxCols, which no
// schema can have a column at.
func TestApplyOpsTrimByCol(t *testing.T) {
	cases := []struct {
		name string
		b    int64
		want uint16
	}{
		{"zero", 0, 0},
		{"in range", 7, 7},
		{"highest addressable", int64(wire.OperateMaxCols) - 1, wire.OperateMaxCols - 1},
		{"negative", -1, wire.OperateMaxCols},
		{"negative aliasing zero", -65536, wire.OperateMaxCols},
		{"one past the top", int64(wire.OperateMaxCols), wire.OperateMaxCols},
		{"aliasing zero", 1 << 16, wire.OperateMaxCols},
		{"aliasing column two", 1<<16 + 2, wire.OperateMaxCols},
		{"max", math.MaxInt64, wire.OperateMaxCols},
		{"min", math.MinInt64, wire.OperateMaxCols},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if got := trimByCol(tc.b); got != tc.want {
				t.Fatalf("trimByCol(%d)=%d, want %d", tc.b, got, tc.want)
			}
			f := newFakeEngine()
			ops := []wire.OperateOp{{Opcode: wire.OperateOpTRIM, Path: fakeFieldPath("tbl"),
				A: 1, Aux: wire.OperatePolicyMinCol, B: tc.b}}
			if _, _, err := applyOps(f, ops, 0); err != nil {
				t.Fatalf("applyOps: %v", err)
			}
			if len(f.trimmed) != 1 || f.trimmed[0].byCol != tc.want {
				t.Fatalf("trim called with %+v, want byCol %d", f.trimmed, tc.want)
			}
		})
	}
}

// TestApplyOpsTrimPolicy: an unknown eviction policy is rejected before
// resolve, the way the oracle rejects it — a malformed op is malformed
// whatever the record holds.
func TestApplyOpsTrimPolicy(t *testing.T) {
	f := newFakeEngine()
	ops := []wire.OperateOp{{Opcode: wire.OperateOpTRIM, Path: fakeFieldPath("tbl"), A: 1,
		Aux: wire.OperatePolicyMaxCol + 1}}
	if _, _, err := applyOps(f, ops, 0); !errors.Is(err, wire.ErrOperateSchema) {
		t.Fatalf("err=%v, want wire.ErrOperateSchema", err)
	}
	for _, call := range f.calls {
		if call == "resolve:"+pathKey(fakeFieldPath("tbl")) {
			t.Fatal("resolve ran before the policy check rejected the op")
		}
	}
	if len(f.trimmed) != 0 {
		t.Fatalf("trim ran: %+v", f.trimmed)
	}
}
