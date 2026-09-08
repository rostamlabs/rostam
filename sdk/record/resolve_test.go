// SPDX-License-Identifier: Apache-2.0

package record

import (
	"bytes"
	"encoding/binary"
	"errors"
	"math"
	"math/rand"
	"sync"
	"testing"

	"github.com/rostamlabs/rostam/sdk/wire"
)

// su64 reinterprets a signed value as the uint64 bit pattern Cell.U stores
// for a signed type. A constant expression like uint64(int64(-5)) does not
// compile, so negative fixtures go through this helper.
func su64(i int64) uint64 { return uint64(i) }

// sinkResult keeps TestResolveNoAlloc's result live so the compiler cannot
// delete the call it is measuring.
var sinkResult Result

// sessionRecord builds the design doc's session record (§2.2 example, plus
// a signed field and a variable-length tail field so the tail walk is
// exercised) and returns its stored bytes:
//
//	#0 rc   U8      = 7
//	#1 bc   U8      = 3
//	#2 hist U32     = 1234
//	#3 bal  I32     = -5
//	#4 tag  BYTES   = "de"
//	#5 b    TABLE(key U64; c0 "hi" U32, c1 "lo" U32) = {42: (500, 1), 99: (7, 2)}
func sessionRecord(t *testing.T, storeNames bool) ([]byte, *wire.Record) {
	t.Helper()
	s := &wire.Schema{
		Version:    3,
		StoreNames: storeNames,
		Fields: []wire.FieldDef{
			{Name: "rc", Type: wire.OperateTypeU8},
			{Name: "bc", Type: wire.OperateTypeU8},
			{Name: "hist", Type: wire.OperateTypeU32},
			{Name: "bal", Type: wire.OperateTypeI32},
			{Name: "tag", Type: wire.OperateTypeBytes},
			{Name: "b", Type: wire.OperateTypeTable, Table: &wire.TableDef{
				KeyType: wire.OperateTypeU64,
				Cols: []wire.ColumnDef{
					{Name: "hi", Type: wire.OperateTypeU32},
					{Name: "lo", Type: wire.OperateTypeU32},
				},
			}},
		},
	}
	if err := s.Validate(); err != nil {
		t.Fatalf("session schema invalid: %v", err)
	}
	rec := &wire.Record{Mode: wire.OperateModeSchema, Schema: s, Fields: []wire.Field{
		{Cell: wire.Cell{Type: wire.OperateTypeU8, U: 7}},
		{Cell: wire.Cell{Type: wire.OperateTypeU8, U: 3}},
		{Cell: wire.Cell{Type: wire.OperateTypeU32, U: 1234}},
		{Cell: wire.Cell{Type: wire.OperateTypeI32, U: su64(-5)}},
		{Cell: wire.Cell{Type: wire.OperateTypeBytes, B: []byte("de")}},
		{Cell: wire.Cell{Type: wire.OperateTypeTable}, Table: &wire.Table{Rows: []wire.Row{
			{Key: leKey(t, 42), Cols: []wire.Col{
				{Cell: wire.Cell{Type: wire.OperateTypeU32, U: 500}},
				{Cell: wire.Cell{Type: wire.OperateTypeU32, U: 1}},
			}},
			{Key: leKey(t, 99), Cols: []wire.Col{
				{Cell: wire.Cell{Type: wire.OperateTypeU32, U: 7}},
				{Cell: wire.Cell{Type: wire.OperateTypeU32, U: 2}},
			}},
		}}},
	}}
	enc := rec.Encode()
	if enc == nil {
		t.Fatal("session record failed to encode")
	}
	dec, err := wire.DecodeRecord(enc)
	if err != nil {
		t.Fatalf("session record failed to decode: %v", err)
	}
	return enc, dec
}

func leKey(t *testing.T, v uint64) []byte {
	t.Helper()
	b := make([]byte, 8)
	binary.LittleEndian.PutUint64(b, v)
	return b
}

// dynamicRecord builds the dynamic-mode example: a scalar, a BYTES field,
// and a table with one 8-byte little-endian key (what client.KeyU64
// produces, addressable in decimal) and one raw ASCII key (addressable
// quoted).
func dynamicRecord(t *testing.T) ([]byte, *wire.Record) {
	t.Helper()
	rec := &wire.Record{Mode: wire.OperateModeDynamic, Fields: []wire.Field{
		{Name: "cnt", Cell: wire.Cell{Type: wire.OperateTypeU32, U: 9}},
		{Name: "raw", Cell: wire.Cell{Type: wire.OperateTypeBytes, B: []byte("hi")}},
		{Name: "t", Cell: wire.Cell{Type: wire.OperateTypeTable}, Table: &wire.Table{Rows: []wire.Row{
			{Key: leKey(t, 42), Cols: []wire.Col{
				{Name: "hi", Cell: wire.Cell{Type: wire.OperateTypeU32, U: 500}},
				{Name: "lo", Cell: wire.Cell{Type: wire.OperateTypeU8, U: 1}},
			}},
			{Key: []byte("DE"), Cols: []wire.Col{
				{Name: "hi", Cell: wire.Cell{Type: wire.OperateTypeU32, U: 7}},
			}},
		}}},
	}}
	enc := rec.Encode()
	if enc == nil {
		t.Fatal("dynamic record failed to encode")
	}
	dec, err := wire.DecodeRecord(enc)
	if err != nil {
		t.Fatalf("dynamic record failed to decode: %v", err)
	}
	return enc, dec
}

// resultsEqual compares two Results structurally, treating two NaN cells as
// equal (reflect.DeepEqual does not).
func resultsEqual(a, b Result) bool {
	if a.Kind != b.Kind || a.Count != b.Count {
		return false
	}
	if a.Cell.Type != b.Cell.Type || a.Cell.N != b.Cell.N || a.Cell.U != b.Cell.U {
		return false
	}
	if !bytes.Equal(a.Cell.B, b.Cell.B) {
		return false
	}
	if math.IsNaN(a.Cell.F) || math.IsNaN(b.Cell.F) {
		return math.IsNaN(a.Cell.F) && math.IsNaN(b.Cell.F)
	}
	return a.Cell.F == b.Cell.F
}

// isZeroResult reports whether res is the zero Result — the only thing an
// error return may carry. Checking Kind alone would not catch it: Absent is
// itself the zero Kind, so a stray Count or Cell would slip through.
func isZeroResult(res Result) bool {
	return res.Kind == Absent && res.Count == 0 &&
		res.Cell.Type == 0 && res.Cell.N == 0 && res.Cell.U == 0 &&
		res.Cell.F == 0 && res.Cell.B == nil
}

// errClass reduces an error to the class the oracle comparison cares about.
func errClass(err error) string {
	switch {
	case err == nil:
		return "nil"
	case errors.Is(err, ErrPath):
		return "path"
	case errors.Is(err, ErrRecord):
		return "record"
	default:
		return "other"
	}
}

func mustPath(t *testing.T, s string) Path {
	t.Helper()
	p, err := ParsePath(s)
	if err != nil {
		t.Fatalf("ParsePath(%q): %v", s, err)
	}
	return p
}

func TestResolveSessionExample(t *testing.T) {
	cases := []struct {
		path    string
		want    Result
		wantErr error
	}{
		{path: "rc", want: Result{Kind: Scalar, Cell: wire.Cell{Type: wire.OperateTypeU8, U: 7}}},
		{path: "bc", want: Result{Kind: Scalar, Cell: wire.Cell{Type: wire.OperateTypeU8, U: 3}}},
		{path: "hist", want: Result{Kind: Scalar, Cell: wire.Cell{Type: wire.OperateTypeU32, U: 1234}}},
		{path: "bal", want: Result{Kind: Scalar, Cell: wire.Cell{Type: wire.OperateTypeI32, U: su64(-5)}}},
		{path: "tag", want: Result{Kind: Scalar, Cell: wire.Cell{Type: wire.OperateTypeBytes, B: []byte("de")}}},
		{path: "b", want: Result{Kind: Table}},
		{path: "b#count", want: Result{Kind: Count, Count: 2}},
		{path: "b/42", want: Result{Kind: RowPresent}},
		{path: "b/99", want: Result{Kind: RowPresent}},
		{path: "b/7", want: Result{Kind: Absent}},
		{path: "b/42/hi", want: Result{Kind: Scalar, Cell: wire.Cell{Type: wire.OperateTypeU32, U: 500}}},
		{path: "b/42/lo", want: Result{Kind: Scalar, Cell: wire.Cell{Type: wire.OperateTypeU32, U: 1}}},
		{path: "b/99/hi", want: Result{Kind: Scalar, Cell: wire.Cell{Type: wire.OperateTypeU32, U: 7}}},
		{path: "b/99/lo", want: Result{Kind: Scalar, Cell: wire.Cell{Type: wire.OperateTypeU32, U: 2}}},
		{path: "b/42/#0", want: Result{Kind: Scalar, Cell: wire.Cell{Type: wire.OperateTypeU32, U: 500}}},
		{path: "b/42/zz", want: Result{Kind: Absent}},
		{path: "b/42/#9", want: Result{Kind: Absent}},
		{path: `b/"quoted"`, want: Result{Kind: Absent}}, // wrong key width for a U64 key
		{path: "b/18446744073709551615", want: Result{Kind: Absent}},
		{path: "zz", want: Result{Kind: Absent}},
		{path: "#0", want: Result{Kind: Scalar, Cell: wire.Cell{Type: wire.OperateTypeU8, U: 7}}},
		{path: "#5#count", want: Result{Kind: Count, Count: 2}},
		{path: "#9", want: Result{Kind: Absent}},
		{path: "#9/42", want: Result{Kind: Absent}},
		{path: "rc#count", wantErr: ErrPath},
		{path: "rc/42", wantErr: ErrPath},
		{path: "rc/42/hi", wantErr: ErrPath},
		{path: "tag/42", wantErr: ErrPath},
	}

	enc, dec := sessionRecord(t, true)
	r := NewResolver(8)
	for _, tc := range cases {
		p := mustPath(t, tc.path)
		got, err := r.Resolve(enc, p)
		if tc.wantErr != nil {
			if !errors.Is(err, tc.wantErr) {
				t.Errorf("Resolve(%q) error = %v, want %v", tc.path, err, tc.wantErr)
			}
		} else {
			if err != nil {
				t.Errorf("Resolve(%q): unexpected error %v", tc.path, err)
				continue
			}
			if !resultsEqual(got, tc.want) {
				t.Errorf("Resolve(%q) = %+v, want %+v", tc.path, got, tc.want)
			}
		}
		// The oracle must agree on every one of these, error class included.
		oracle, oerr := resolveTree(dec, p)
		if errClass(err) != errClass(oerr) {
			t.Errorf("Resolve(%q) error class %s, oracle %s", tc.path, errClass(err), errClass(oerr))
		}
		if err == nil && oerr == nil && !resultsEqual(got, oracle) {
			t.Errorf("Resolve(%q) = %+v, oracle = %+v", tc.path, got, oracle)
		}
	}
}

func TestResolveSessionExampleWithoutNames(t *testing.T) {
	cases := []struct {
		path    string
		want    Result
		wantErr error
	}{
		{path: "#0", want: Result{Kind: Scalar, Cell: wire.Cell{Type: wire.OperateTypeU8, U: 7}}},
		{path: "#3", want: Result{Kind: Scalar, Cell: wire.Cell{Type: wire.OperateTypeI32, U: su64(-5)}}},
		{path: "#4", want: Result{Kind: Scalar, Cell: wire.Cell{Type: wire.OperateTypeBytes, B: []byte("de")}}},
		{path: "#5", want: Result{Kind: Table}},
		{path: "#5#count", want: Result{Kind: Count, Count: 2}},
		{path: "#5/42", want: Result{Kind: RowPresent}},
		{path: "#5/42/#0", want: Result{Kind: Scalar, Cell: wire.Cell{Type: wire.OperateTypeU32, U: 500}}},
		{path: "#5/42/#1", want: Result{Kind: Scalar, Cell: wire.Cell{Type: wire.OperateTypeU32, U: 1}}},
		{path: "#5/7", want: Result{Kind: Absent}},
		// Names cannot apply at all when the schema stores none.
		{path: "rc", wantErr: ErrPath},
		{path: "b#count", wantErr: ErrPath},
		{path: "#5/42/hi", wantErr: ErrPath},
	}

	enc, dec := sessionRecord(t, false)
	r := NewResolver(8)
	for _, tc := range cases {
		p := mustPath(t, tc.path)
		got, err := r.Resolve(enc, p)
		if tc.wantErr != nil {
			if !errors.Is(err, tc.wantErr) {
				t.Errorf("Resolve(%q) error = %v, want %v", tc.path, err, tc.wantErr)
			}
		} else {
			if err != nil {
				t.Errorf("Resolve(%q): unexpected error %v", tc.path, err)
				continue
			}
			if !resultsEqual(got, tc.want) {
				t.Errorf("Resolve(%q) = %+v, want %+v", tc.path, got, tc.want)
			}
		}
		oracle, oerr := resolveTree(dec, p)
		if errClass(err) != errClass(oerr) {
			t.Errorf("Resolve(%q) error class %s, oracle %s", tc.path, errClass(err), errClass(oerr))
		}
		if err == nil && oerr == nil && !resultsEqual(got, oracle) {
			t.Errorf("Resolve(%q) = %+v, oracle = %+v", tc.path, got, oracle)
		}
	}
}

func TestResolveDynamicExample(t *testing.T) {
	cases := []struct {
		path    string
		want    Result
		wantErr error
	}{
		{path: "cnt", want: Result{Kind: Scalar, Cell: wire.Cell{Type: wire.OperateTypeU32, U: 9}}},
		{path: "raw", want: Result{Kind: Scalar, Cell: wire.Cell{Type: wire.OperateTypeBytes, B: []byte("hi")}}},
		{path: "t", want: Result{Kind: Table}},
		{path: "t#count", want: Result{Kind: Count, Count: 2}},
		// A decimal row key addresses the 8-byte little-endian encoding of
		// the number — what client.KeyU64 writes.
		{path: "t/42", want: Result{Kind: RowPresent}},
		{path: "t/42/hi", want: Result{Kind: Scalar, Cell: wire.Cell{Type: wire.OperateTypeU32, U: 500}}},
		{path: "t/42/lo", want: Result{Kind: Scalar, Cell: wire.Cell{Type: wire.OperateTypeU8, U: 1}}},
		{path: "t/7", want: Result{Kind: Absent}},
		// A quoted row key addresses raw bytes.
		{path: `t/"DE"`, want: Result{Kind: RowPresent}},
		{path: `t/"DE"/hi`, want: Result{Kind: Scalar, Cell: wire.Cell{Type: wire.OperateTypeU32, U: 7}}},
		{path: `t/"DE"/lo`, want: Result{Kind: Absent}},
		{path: `t/"XX"`, want: Result{Kind: Absent}},
		{path: "zz", want: Result{Kind: Absent}},
		{path: "zz/1/c", want: Result{Kind: Absent}},
		// Positions never apply in dynamic mode.
		{path: "#0", wantErr: ErrPath},
		{path: "#0#count", wantErr: ErrPath},
		{path: "t/42/#0", wantErr: ErrPath},
		// Shape errors.
		{path: "cnt#count", wantErr: ErrPath},
		{path: "cnt/42", wantErr: ErrPath},
	}

	enc, dec := dynamicRecord(t)
	r := NewResolver(8)
	for _, tc := range cases {
		p := mustPath(t, tc.path)
		got, err := r.Resolve(enc, p)
		if tc.wantErr != nil {
			if !errors.Is(err, tc.wantErr) {
				t.Errorf("Resolve(%q) error = %v, want %v", tc.path, err, tc.wantErr)
			}
		} else {
			if err != nil {
				t.Errorf("Resolve(%q): unexpected error %v", tc.path, err)
				continue
			}
			if !resultsEqual(got, tc.want) {
				t.Errorf("Resolve(%q) = %+v, want %+v", tc.path, got, tc.want)
			}
		}
		oracle, oerr := resolveTree(dec, p)
		if errClass(err) != errClass(oerr) {
			t.Errorf("Resolve(%q) error class %s, oracle %s", tc.path, errClass(err), errClass(oerr))
		}
		if err == nil && oerr == nil && !resultsEqual(got, oracle) {
			t.Errorf("Resolve(%q) = %+v, oracle = %+v", tc.path, got, oracle)
		}
	}
}

// TestResolveMatchesOracle is the property this whole task exists to hold:
// over random records in both modes and random paths, the byte-level
// resolver returns exactly what the tree oracle returns — same Result, or
// the same error class.
func TestResolveMatchesOracle(t *testing.T) {
	records, paths := 2000, 20
	if testing.Short() {
		records, paths = 200, 8
	}
	rng := rand.New(rand.NewSource(20260908))
	r := NewResolver(64)
	checked, specials := 0, 0
	seen := map[string]int{}
	for i := 0; i < records; i++ {
		var enc []byte
		var dec *wire.Record
		var err error
		if i%2 == 0 {
			enc, dec, err = randomSchemaRecord(rng)
		} else {
			enc, dec, err = randomDynamicRecord(rng)
		}
		if err != nil {
			t.Fatalf("generator: %v", err)
		}
		for j := 0; j < paths; j++ {
			s := randomPath(rng, dec)
			p, perr := ParsePath(s)
			if perr != nil {
				continue
			}
			checked++
			got, gerr := r.Resolve(enc, p)
			want, werr := resolveTree(dec, p)
			if errClass(gerr) == "record" {
				t.Fatalf("record %d path %q: resolver called a well-formed record malformed: %v", i, s, gerr)
			}
			if errClass(gerr) != errClass(werr) {
				t.Fatalf("record %d path %q: error class %s (%v), oracle %s (%v)", i, s, errClass(gerr), gerr, errClass(werr), werr)
			}
			if gerr == nil && !resultsEqual(got, want) {
				t.Fatalf("record %d path %q: got %+v, oracle %+v", i, s, got, want)
			}
			seen[outcomeName(got, gerr, dec.Mode)]++
			if gerr == nil && got.Kind == Scalar &&
				(math.IsNaN(got.Cell.F) || math.IsInf(got.Cell.F, 0)) {
				specials++
			}
		}
	}
	if checked < records {
		t.Fatalf("only %d path resolutions checked over %d records", checked, records)
	}
	t.Logf("%d resolutions checked (%d NaN/Inf scalars): %v", checked, specials, seen)
	// NaN and the infinities are the float values the resolver treats
	// specially (readCellAt rejects every NaN spelling but the canonical
	// one), so at least some must actually reach the oracle comparison.
	if specials == 0 {
		t.Fatalf("no NaN or Inf float cell was ever resolved; the generator stopped drawing them")
	}
	// A property test that only ever produced "absent" would pass while
	// proving nothing, so require every outcome, in both modes, to show up.
	floor := checked / 200
	if floor < 1 {
		floor = 1
	}
	for _, mode := range []string{"schema", "dynamic"} {
		for _, kind := range []string{"scalar", "count", "row_present", "table", "absent", "path_error"} {
			if n := seen[mode+"/"+kind]; n < floor {
				t.Fatalf("only %d %s %s outcomes over %d resolutions (want >= %d): %v",
					n, mode, kind, checked, floor, seen)
			}
		}
	}
}

// outcomeName labels one resolution for the distribution check above.
func outcomeName(res Result, err error, mode uint8) string {
	m := "dynamic"
	if mode == wire.OperateModeSchema {
		m = "schema"
	}
	if err != nil {
		return m + "/" + errClass(err) + "_error"
	}
	switch res.Kind {
	case Scalar:
		return m + "/scalar"
	case Count:
		return m + "/count"
	case RowPresent:
		return m + "/row_present"
	case Table:
		return m + "/table"
	default:
		return m + "/absent"
	}
}

// TestResolveHostileNeverPanics drives the resolver with every prefix and
// every single-byte mutation of both example records, plus random bytes: a
// stored record is attacker-influenced (set_payload stores whatever a
// client sends), so no input may panic, produce an error outside the
// package's two classes, or mutate the caller's slice.
func TestResolveHostileNeverPanics(t *testing.T) {
	sess, _ := sessionRecord(t, true)
	dyn, _ := dynamicRecord(t)

	paths := []Path{
		mustPath(t, "rc"),
		mustPath(t, "tag"),
		mustPath(t, "b#count"),
		mustPath(t, "b/42"),
		mustPath(t, "b/42/hi"),
		mustPath(t, "#5/42/#1"),
		mustPath(t, "cnt"),
		mustPath(t, "t#count"),
		mustPath(t, `t/"DE"/hi`),
		mustPath(t, "t/42/lo"),
	}
	r := NewResolver(16)

	run := func(name string, in []byte) {
		before := append([]byte(nil), in...)
		for _, p := range paths {
			res, err := r.Resolve(in, p)
			if err != nil && errClass(err) == "other" {
				t.Fatalf("%s: error outside ErrPath/ErrRecord: %v", name, err)
			}
			if err != nil && !isZeroResult(res) {
				t.Fatalf("%s: error %v returned a non-zero result %+v", name, err, res)
			}
		}
		if !bytes.Equal(before, in) {
			t.Fatalf("%s: Resolve mutated its input", name)
		}
	}

	for _, base := range [][]byte{sess, dyn} {
		for n := 0; n <= len(base); n++ {
			run("prefix", base[:n:n])
		}
		mut := append([]byte(nil), base...)
		for i := range base {
			orig := mut[i]
			for v := 0; v < 256; v++ {
				mut[i] = byte(v)
				run("mutation", mut)
			}
			mut[i] = orig
		}
	}

	rng := rand.New(rand.NewSource(11))
	n := 4000
	if testing.Short() {
		n = 500
	}
	for i := 0; i < n; i++ {
		buf := make([]byte, rng.Intn(40))
		for j := range buf {
			buf[j] = byte(rng.Intn(256))
		}
		run("random", buf)
	}
}

// TestResolveCellDoesNotAliasInput guards the hazard a byte-level resolver
// invites: a BYTES or FIXED cell must be copied out of the record, never
// handed back as a slice into the caller's (possibly reused) buffer.
func TestResolveCellDoesNotAliasInput(t *testing.T) {
	enc, _ := sessionRecord(t, true)
	buf := append([]byte(nil), enc...)
	r := NewResolver(4)
	res, err := r.Resolve(buf, mustPath(t, "tag"))
	if err != nil {
		t.Fatalf("Resolve: %v", err)
	}
	if string(res.Cell.B) != "de" {
		t.Fatalf("tag = %q, want %q", res.Cell.B, "de")
	}
	for i := range buf {
		buf[i] ^= 0xFF
	}
	if string(res.Cell.B) != "de" {
		t.Fatalf("cell aliased the record bytes: tag became %q", res.Cell.B)
	}
}

// TestResolveNoAlloc is the point of resolving over bytes instead of
// decoding the tree: once the schema is cached, reading a column of an
// existing row allocates nothing.
func TestResolveNoAlloc(t *testing.T) {
	enc, _ := sessionRecord(t, true)
	r := NewResolver(8)
	p := mustPath(t, "b/42/hi")
	if _, err := r.Resolve(enc, p); err != nil {
		t.Fatalf("warm-up Resolve: %v", err)
	}
	if got := testing.AllocsPerRun(200, func() {
		res, err := r.Resolve(enc, p)
		if err != nil {
			t.Fatalf("Resolve: %v", err)
		}
		sinkResult = res
	}); got > 0 {
		t.Fatalf("Resolve allocated %v times per run, want 0", got)
	}
	if sinkResult.Kind != Scalar || sinkResult.Cell.U != 500 {
		t.Fatalf("measured the wrong thing: %+v", sinkResult)
	}
}

// TestSchemaCacheBounded checks the cache honours its bound (clear when
// full) and that a hit and a miss are indistinguishable in the answer.
func TestSchemaCacheBounded(t *testing.T) {
	r := NewResolver(2)
	var encs [][]byte
	for v := 0; v < 5; v++ {
		s := &wire.Schema{Version: uint16(v), StoreNames: true, Fields: []wire.FieldDef{
			{Name: "rc", Type: wire.OperateTypeU32},
		}}
		rec := &wire.Record{Mode: wire.OperateModeSchema, Schema: s, Fields: []wire.Field{
			{Cell: wire.Cell{Type: wire.OperateTypeU32, U: uint64(100 + v)}},
		}}
		enc := rec.Encode()
		if enc == nil {
			t.Fatalf("schema version %d failed to encode", v)
		}
		encs = append(encs, enc)
	}

	p := mustPath(t, "rc")
	for round := 0; round < 3; round++ {
		for v, enc := range encs {
			got, err := r.Resolve(enc, p)
			if err != nil {
				t.Fatalf("Resolve(version %d): %v", v, err)
			}
			if got.Kind != Scalar || got.Cell.U != uint64(100+v) {
				t.Fatalf("Resolve(version %d) = %+v", v, got)
			}
			r.mu.RLock()
			size := len(r.cache)
			r.mu.RUnlock()
			if size > 2 {
				t.Fatalf("cache grew to %d entries, bound is 2", size)
			}
		}
	}
}

// TestResolverConcurrent backs the claim Task 5 relies on: one Resolver is
// shared by every query goroutine, so its schema cache must tolerate
// concurrent hits, misses, and the clear-when-full it does under the write
// lock. Run under -race this is the test that would catch a missing lock.
func TestResolverConcurrent(t *testing.T) {
	r := NewResolver(2) // small enough that the workers keep evicting
	sess, _ := sessionRecord(t, true)
	dyn, _ := dynamicRecord(t)
	var extra [][]byte
	for v := 0; v < 4; v++ {
		s := &wire.Schema{Version: uint16(v), StoreNames: true, Fields: []wire.FieldDef{
			{Name: "rc", Type: wire.OperateTypeU32},
		}}
		rec := &wire.Record{Mode: wire.OperateModeSchema, Schema: s, Fields: []wire.Field{
			{Cell: wire.Cell{Type: wire.OperateTypeU32, U: uint64(100 + v)}},
		}}
		extra = append(extra, rec.Encode())
	}

	type job struct {
		rec  []byte
		path string
		want Result
	}
	jobs := []job{
		{sess, "b/42/hi", Result{Kind: Scalar, Cell: wire.Cell{Type: wire.OperateTypeU32, U: 500}}},
		{sess, "b#count", Result{Kind: Count, Count: 2}},
		{dyn, `t/"DE"/hi`, Result{Kind: Scalar, Cell: wire.Cell{Type: wire.OperateTypeU32, U: 7}}},
	}
	for v, enc := range extra {
		jobs = append(jobs, job{enc, "rc", Result{Kind: Scalar, Cell: wire.Cell{Type: wire.OperateTypeU32, U: uint64(100 + v)}}})
	}
	paths := make([]Path, len(jobs))
	for i, j := range jobs {
		paths[i] = mustPath(t, j.path)
	}

	var wg sync.WaitGroup
	for w := 0; w < 8; w++ {
		wg.Add(1)
		go func(w int) {
			defer wg.Done()
			for i := 0; i < 300; i++ {
				k := (i + w) % len(jobs)
				got, err := r.Resolve(jobs[k].rec, paths[k])
				if err != nil {
					t.Errorf("worker %d: Resolve(%q): %v", w, jobs[k].path, err)
					return
				}
				if !resultsEqual(got, jobs[k].want) {
					t.Errorf("worker %d: Resolve(%q) = %+v, want %+v", w, jobs[k].path, got, jobs[k].want)
					return
				}
			}
		}(w)
	}
	wg.Wait()
}

// TestResolveWithoutCache checks a resolver built with no cache budget
// still answers, and answers identically to a cached one — the cache may
// never influence the result.
func TestResolveWithoutCache(t *testing.T) {
	enc, _ := sessionRecord(t, true)
	uncached := NewResolver(0)
	cached := NewResolver(8)
	for _, s := range []string{"rc", "tag", "b#count", "b/42/hi", "b/7"} {
		p := mustPath(t, s)
		a, aerr := uncached.Resolve(enc, p)
		b, berr := cached.Resolve(enc, p)
		if errClass(aerr) != errClass(berr) {
			t.Fatalf("%q: uncached err %v, cached err %v", s, aerr, berr)
		}
		if !resultsEqual(a, b) {
			t.Fatalf("%q: uncached %+v, cached %+v", s, a, b)
		}
	}
}

// TestResolveWideFixedKey pins the case a uint64 cannot hold: a FIXED row
// key wider than 8 bytes, where a decimal segment denotes the number's
// little-endian bytes zero-filled out to the key width. The randomised
// oracle test draws these too; this one names the arithmetic so a
// regression in it reads as itself.
func TestResolveWideFixedKey(t *testing.T) {
	const kw = 12
	wide := func(v uint64) []byte {
		b := make([]byte, kw)
		binary.LittleEndian.PutUint64(b[:8], v)
		return b
	}
	s := &wire.Schema{Version: 1, StoreNames: true, Fields: []wire.FieldDef{
		{Name: "b", Type: wire.OperateTypeTable, Table: &wire.TableDef{
			KeyType: wire.OperateTypeFixed,
			KeyN:    kw,
			Cols:    []wire.ColumnDef{{Name: "hi", Type: wire.OperateTypeU32}},
		}},
	}}
	if err := s.Validate(); err != nil {
		t.Fatalf("schema invalid: %v", err)
	}
	// "zzzz…" sorts after every little-endian small number, so the rows are
	// already in the bytewise order a FIXED key is stored in.
	textKey := bytes.Repeat([]byte("z"), kw)
	rec := &wire.Record{Mode: wire.OperateModeSchema, Schema: s, Fields: []wire.Field{
		{Cell: wire.Cell{Type: wire.OperateTypeTable}, Table: &wire.Table{Rows: []wire.Row{
			{Key: wide(42), Cols: []wire.Col{{Cell: wire.Cell{Type: wire.OperateTypeU32, U: 500}}}},
			{Key: wide(300), Cols: []wire.Col{{Cell: wire.Cell{Type: wire.OperateTypeU32, U: 7}}}},
			{Key: textKey, Cols: []wire.Col{{Cell: wire.Cell{Type: wire.OperateTypeU32, U: 9}}}},
		}}},
	}}
	enc := rec.Encode()
	if enc == nil {
		t.Fatal("record failed to encode")
	}
	dec, err := wire.DecodeRecord(enc)
	if err != nil {
		t.Fatalf("record failed to decode: %v", err)
	}

	cases := []struct {
		path string
		want Result
	}{
		{"b/42", Result{Kind: RowPresent}},
		{"b/42/hi", Result{Kind: Scalar, Cell: wire.Cell{Type: wire.OperateTypeU32, U: 500}}},
		{"b/300/hi", Result{Kind: Scalar, Cell: wire.Cell{Type: wire.OperateTypeU32, U: 7}}},
		{`b/"zzzzzzzzzzzz"/hi`, Result{Kind: Scalar, Cell: wire.Cell{Type: wire.OperateTypeU32, U: 9}}},
		{"b/7", Result{Kind: Absent}},
		{"b/18446744073709551615", Result{Kind: Absent}},
		{`b/"zzz"`, Result{Kind: Absent}}, // wrong width for this key
		{"b#count", Result{Kind: Count, Count: 3}},
	}
	r := NewResolver(4)
	for _, tc := range cases {
		p := mustPath(t, tc.path)
		got, err := r.Resolve(enc, p)
		if err != nil {
			t.Errorf("Resolve(%q): %v", tc.path, err)
			continue
		}
		if !resultsEqual(got, tc.want) {
			t.Errorf("Resolve(%q) = %+v, want %+v", tc.path, got, tc.want)
		}
		oracle, oerr := resolveTree(dec, p)
		if oerr != nil || !resultsEqual(got, oracle) {
			t.Errorf("Resolve(%q) = %+v, oracle = %+v (err %v)", tc.path, got, oracle, oerr)
		}
	}
}

// TestResolveRejectsMalformedPaths covers the shapes ParsePath never
// builds but a caller assembling a Path by hand could: Resolve is exported
// and Path is a plain struct, so the segment kinds have to be checked
// rather than assumed.
func TestResolveRejectsMalformedPaths(t *testing.T) {
	enc, _ := sessionRecord(t, true)
	r := NewResolver(4)
	cases := []struct {
		name string
		p    Path
	}{
		{"no segments", Path{}},
		{"four segments", Path{Segs: []Segment{
			{Kind: SegField, Name: "b"}, {Kind: SegRow, KeyText: "42"},
			{Kind: SegCol, Name: "hi"}, {Kind: SegCol, Name: "lo"},
		}}},
		{"row first", Path{Segs: []Segment{{Kind: SegRow, KeyText: "42"}}}},
		{"count with a row after it", Path{Segs: []Segment{
			{Kind: SegCount, Name: "b"}, {Kind: SegRow, KeyText: "42"},
		}}},
		{"field where a row belongs", Path{Segs: []Segment{
			{Kind: SegField, Name: "b"}, {Kind: SegField, Name: "hi"},
		}}},
		{"row where a column belongs", Path{Segs: []Segment{
			{Kind: SegField, Name: "b"}, {Kind: SegRow, KeyText: "42"}, {Kind: SegRow, KeyText: "1"},
		}}},
	}
	for _, tc := range cases {
		if _, err := r.Resolve(enc, tc.p); !errors.Is(err, ErrPath) {
			t.Errorf("%s: error = %v, want ErrPath", tc.name, err)
		}
	}
}

// TestSchemaBlobLenAgreesWithDecodeSchema pins the cache's fast path to the
// authoritative decoder: the scanner that finds where a schema blob ends
// must agree with DecodeSchema on every blob DecodeSchema accepts. A
// disagreement only costs a cache miss (the cached key is compared byte for
// byte), but a silent drift would quietly disable the cache.
func TestSchemaBlobLenAgreesWithDecodeSchema(t *testing.T) {
	rng := rand.New(rand.NewSource(3))
	for i := 0; i < 500; i++ {
		enc, _, err := randomSchemaRecord(rng)
		if err != nil {
			t.Fatalf("generator: %v", err)
		}
		blob := enc[1:]
		_, want, derr := wire.DecodeSchema(blob)
		if derr != nil {
			t.Fatalf("generated schema failed to decode: %v", derr)
		}
		got, ok := schemaBlobLen(blob)
		if !ok {
			t.Fatalf("record %d: schemaBlobLen refused a valid blob", i)
		}
		if got != want {
			t.Fatalf("record %d: schemaBlobLen = %d, DecodeSchema consumed %d", i, got, want)
		}
	}
}
