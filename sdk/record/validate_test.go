// SPDX-License-Identifier: Apache-2.0

package record

import (
	"bytes"
	"errors"
	"math/rand"
	"testing"

	"github.com/rostamlabs/rostam/sdk/wire"
)

// TestValidateAcceptsBothModes pins the positive half: the two canonical
// fixtures the resolver is tested against — a schema-mode record with and
// without stored names, and a dynamic-mode one — are records the operate
// engines open, so the ingest gate must let them through.
func TestValidateAcceptsBothModes(t *testing.T) {
	named, _ := sessionRecord(t, true)
	nameless, _ := sessionRecord(t, false)
	dyn, _ := dynamicRecord(t)

	for _, c := range []struct {
		name string
		rec  []byte
	}{
		{"schema mode, names stored", named},
		{"schema mode, names omitted", nameless},
		{"dynamic mode", dyn},
	} {
		t.Run(c.name, func(t *testing.T) {
			before := append([]byte(nil), c.rec...)
			if err := Validate(c.rec); err != nil {
				t.Fatalf("Validate = %v, want nil", err)
			}
			if !bytes.Equal(before, c.rec) {
				t.Fatal("Validate mutated its input")
			}
		})
	}
}

// TestValidateRejectsEmpty covers the shapes that carry no record at all: nil,
// empty, a bare mode byte with nothing behind it, and the two mode bytes no
// record uses. Each must be an error wrapping ErrRecord — never a panic, and
// never a silent accept, because these are exactly the bytes a caller reaches
// the ingest path with when it stores a truncated or foreign value.
func TestValidateRejectsEmpty(t *testing.T) {
	for _, c := range []struct {
		name string
		rec  []byte
	}{
		{"nil", nil},
		{"empty", []byte{}},
		{"bare schema mode byte", []byte{byte(wire.OperateModeSchema)}},
		{"bare dynamic mode byte", []byte{byte(wire.OperateModeDynamic)}},
		{"mode byte 0", []byte{0x00, 0x01}},
		{"mode byte past the enum", []byte{0xFF, 0x01}},
	} {
		t.Run(c.name, func(t *testing.T) {
			err := Validate(c.rec)
			if err == nil {
				t.Fatal("Validate = nil, want an error")
			}
			if !errors.Is(err, ErrRecord) {
				t.Fatalf("Validate = %v, want an error wrapping ErrRecord", err)
			}
		})
	}
}

// TestValidateRejectsHostileBytes drives every single-byte mutation and every
// prefix of both canonical fixtures through Validate. The only two acceptable
// outcomes are "nil" and "an error wrapping ErrRecord": a panic inside the
// ingest gate would be reachable from any client that stores a payload, and an
// error that does NOT wrap ErrRecord would escape the caller's errors.Is test
// and be reported as an internal fault rather than a bad request.
func TestValidateRejectsHostileBytes(t *testing.T) {
	sess, _ := sessionRecord(t, true)
	dyn, _ := dynamicRecord(t)

	run := func(name string, in []byte) {
		before := append([]byte(nil), in...)
		err := Validate(in)
		if err != nil && !errors.Is(err, ErrRecord) {
			t.Fatalf("%s: error outside ErrRecord: %v", name, err)
		}
		if !bytes.Equal(before, in) {
			t.Fatalf("%s: Validate mutated its input", name)
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

	rng := rand.New(rand.NewSource(23))
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

// TestValidateMatchesDecodeRecord is the equivalence the doc comment claims:
// Validate's acceptance set is EXACTLY wire.DecodeRecord's, over the random
// record generators the resolver oracle uses and over damaged versions of what
// they produce. If Validate ever grows a second, hand-written walk, this is the
// test that catches it diverging — and a divergence is the whole poison class
// phase 1 had to fail closed around: two readers disagreeing about one record.
func TestValidateMatchesDecodeRecord(t *testing.T) {
	rng := rand.New(rand.NewSource(4242))
	iters := 400
	if testing.Short() {
		iters = 60
	}
	check := func(b []byte) {
		t.Helper()
		_, decErr := wire.DecodeRecord(b)
		valErr := Validate(b)
		if (decErr == nil) != (valErr == nil) {
			t.Fatalf("DecodeRecord err = %v but Validate err = %v on % x", decErr, valErr, b)
		}
		if valErr != nil && !errors.Is(valErr, ErrRecord) {
			t.Fatalf("Validate = %v, want an error wrapping ErrRecord", valErr)
		}
	}
	for i := 0; i < iters; i++ {
		var enc []byte
		var err error
		if i%2 == 0 {
			enc, _, err = randomSchemaRecord(rng)
		} else {
			enc, _, err = randomDynamicRecord(rng)
		}
		if err != nil {
			continue // the generator drew a shape the encoder rejects
		}
		check(enc)
		if len(enc) == 0 {
			continue
		}
		// Damage it three ways: truncate, extend, and flip one byte. All three
		// are shapes a torn or forged payload value really arrives in.
		check(enc[:rng.Intn(len(enc)):len(enc)])
		check(append(append([]byte(nil), enc...), byte(rng.Intn(256))))
		mut := append([]byte(nil), enc...)
		mut[rng.Intn(len(mut))] ^= byte(1 + rng.Intn(255))
		check(mut)
	}
}

// BenchmarkValidate reports what the ingest gate costs per record value: the
// session fixture (a small schema-mode record with a two-row table) and a
// table-heavy 64 KiB one, since DecodeRecord's cost is dominated by the rows
// it materialises.
func BenchmarkValidate(b *testing.B) {
	small := benchSessionRecord(b)
	large := benchTableRecord(b, 64<<10)

	for _, c := range []struct {
		name string
		rec  []byte
	}{
		{"session", small},
		{"table-64KiB", large},
	} {
		b.Run(c.name, func(b *testing.B) {
			b.ReportAllocs()
			b.SetBytes(int64(len(c.rec)))
			for i := 0; i < b.N; i++ {
				if err := Validate(c.rec); err != nil {
					b.Fatal(err)
				}
			}
		})
	}
}

// benchSessionRecord is sessionRecord for a benchmark (which has no *testing.T).
func benchSessionRecord(b *testing.B) []byte {
	b.Helper()
	s := &wire.Schema{Version: 3, StoreNames: true, Fields: []wire.FieldDef{
		{Name: "rc", Type: wire.OperateTypeU8},
		{Name: "bal", Type: wire.OperateTypeI32},
		{Name: "tag", Type: wire.OperateTypeBytes},
	}}
	if err := s.Validate(); err != nil {
		b.Fatalf("schema invalid: %v", err)
	}
	enc := (&wire.Record{Mode: wire.OperateModeSchema, Schema: s, Fields: []wire.Field{
		{Cell: wire.Cell{Type: wire.OperateTypeU8, U: 7}},
		{Cell: wire.Cell{Type: wire.OperateTypeI32, U: su64(-5)}},
		{Cell: wire.Cell{Type: wire.OperateTypeBytes, B: []byte("de")}},
	}}).Encode()
	if enc == nil {
		b.Fatal("bench session record failed to encode")
	}
	return enc
}

// benchTableRecord builds a schema-mode record whose single TABLE field holds
// enough rows to reach roughly size bytes — the shape whose decode cost grows,
// so the benchmark shows the gate's cost as a function of record size.
func benchTableRecord(b *testing.B, size int) []byte {
	b.Helper()
	s := &wire.Schema{Version: 1, StoreNames: true, Fields: []wire.FieldDef{
		{Name: "t", Type: wire.OperateTypeTable, Table: &wire.TableDef{
			KeyType: wire.OperateTypeU64,
			Cols: []wire.ColumnDef{
				{Name: "hi", Type: wire.OperateTypeU32},
				{Name: "lo", Type: wire.OperateTypeU32},
			},
		}},
	}}
	if err := s.Validate(); err != nil {
		b.Fatalf("schema invalid: %v", err)
	}
	const rowWidth = 16 // 8-byte key + two U32 columns
	rows := make([]wire.Row, 0, size/rowWidth)
	for i := 0; i < size/rowWidth; i++ {
		key := make([]byte, 8)
		for j := 0; j < 8; j++ {
			key[j] = byte(uint64(i) >> (8 * j)) //nolint:gosec // bounded loop counter
		}
		rows = append(rows, wire.Row{Key: key, Cols: []wire.Col{
			{Cell: wire.Cell{Type: wire.OperateTypeU32, U: uint64(i)}},     //nolint:gosec // bounded
			{Cell: wire.Cell{Type: wire.OperateTypeU32, U: uint64(i) + 3}}, //nolint:gosec // bounded
		}})
	}
	enc := (&wire.Record{Mode: wire.OperateModeSchema, Schema: s, Fields: []wire.Field{
		{Cell: wire.Cell{Type: wire.OperateTypeTable}, Table: &wire.Table{Rows: rows}},
	}}).Encode()
	if enc == nil {
		b.Fatal("bench table record failed to encode")
	}
	return enc
}
