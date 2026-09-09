// SPDX-License-Identifier: Apache-2.0

package rostam

import (
	"context"
	"encoding/binary"
	"math/rand"
	"testing"

	"github.com/rostamlabs/rostam/client"
	"github.com/rostamlabs/rostam/sdk/vtypes"
	"github.com/rostamlabs/rostam/sdk/wire"
	"github.com/rostamlabs/rostam/vector"
)

// su64 reinterprets a signed value as the uint64 bit pattern Cell.U stores for
// a signed type. A constant expression like uint64(int64(-5)) does not compile
// (constant overflow), so negative fixtures go through this helper — per the
// phase-2 constraints, one su64 per package (vector/record_filter_test.go:17,
// ops/operate_oracle_test.go).
func su64(i int64) uint64 { return uint64(i) }

// sessionRecordBytesRC mirrors vector/record_filter_test.go's helper of the
// same name (design doc §4's session-record worked example): a schema-mode
// record with StoreNames set, fields rc/bc/hist/bal/tag/b, rc caller-chosen.
// Duplicated here because vector's test helpers are unexported and this file
// lives in a different package — the same accepted precedent as
// ops/record_malformed_test.go's fixture (phase-2 ledger, Task 5: "the
// duplicated session fixture in ops tests accepted; vector test helpers are
// not importable").
func sessionRecordBytesRC(t *testing.T, rc uint8) []byte {
	t.Helper()
	leKey := func(v uint64) []byte {
		b := make([]byte, 8)
		binary.LittleEndian.PutUint64(b, v)
		return b
	}
	s := &wire.Schema{
		Version:    1,
		StoreNames: true,
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
		{Cell: wire.Cell{Type: wire.OperateTypeU8, U: uint64(rc)}},
		{Cell: wire.Cell{Type: wire.OperateTypeU8, U: 3}},
		{Cell: wire.Cell{Type: wire.OperateTypeU32, U: 1234}},
		{Cell: wire.Cell{Type: wire.OperateTypeI32, U: su64(-5)}},
		{Cell: wire.Cell{Type: wire.OperateTypeBytes, B: []byte("de")}},
		{Cell: wire.Cell{Type: wire.OperateTypeTable}, Table: &wire.Table{Rows: []wire.Row{
			{Key: leKey(42), Cols: []wire.Col{
				{Cell: wire.Cell{Type: wire.OperateTypeU32, U: 500}},
				{Cell: wire.Cell{Type: wire.OperateTypeU32, U: 1}},
			}},
			{Key: leKey(99), Cols: []wire.Col{
				{Cell: wire.Cell{Type: wire.OperateTypeU32, U: 7}},
				{Cell: wire.Cell{Type: wire.OperateTypeU32, U: 2}},
			}},
		}}},
	}}
	enc := rec.Encode()
	if enc == nil {
		t.Fatal("session record failed to encode")
	}
	return enc
}

// TestOperateEndToEndKeepsTheIndexExact is the full-stack invariant behind
// Collection.Operate: after a real mutation applied through the typed client,
// the collection's accelerated record-path filter must answer off the
// MUTATED value, not the value the point was inserted with. It is phase 1's
// TestRecordFilterFirstEqualsBruteForce shape
// (vector/payload_index_record_test.go), run after mutations rather than
// after inserts, and driven end-to-end through a real server via the typed Go
// client rather than the bare engine.
//
// Every point is inserted with rc=0 (below the threshold the filter below
// tests) and then moved, via one Collection.Operate SET per point, to a
// value spread across 0..19 — straddling the threshold. If the accelerated
// filter path answered off the stale inserted value instead of the mutation,
// this would surface as either a missing true match or a false one, not mere
// noise.
func TestOperateEndToEndKeepsTheIndexExact(t *testing.T) {
	const n = 200
	const threshold = 9

	col, cleanup := mustCollection(t)
	defer cleanup()
	ctx := context.Background()

	rng := rand.New(rand.NewSource(11))
	finalRC := make(map[uint64]uint8, n)

	for i := 1; i <= n; i++ {
		id := uint64(i) //nolint:gosec // bounded by n
		vec := []float32{rng.Float32()*2 - 1, rng.Float32()*2 - 1, rng.Float32()*2 - 1, rng.Float32()*2 - 1}
		if err := col.Upsert(ctx, client.WriteRequest{
			ID: id, Vector: vec,
			Metadata: vtypes.Metadata{"session": vtypes.NewRecord(sessionRecordBytesRC(t, 0))},
		}); err != nil {
			t.Fatalf("Upsert %d: %v", id, err)
		}

		target := uint8(i % 20) //nolint:gosec // bounded by 20
		a, err := client.NewOperate(nil).
			Set(client.F("rc"), int64(target)).
			Return(client.F("rc")).
			Args()
		if err != nil {
			t.Fatalf("build operate args for %d: %v", id, err)
		}
		found, res, err := col.Operate(ctx, client.OperateRequest{ID: id, PayloadKey: "session", Args: a})
		if err != nil {
			t.Fatalf("Operate %d: %v", id, err)
		}
		if !found {
			t.Fatalf("Operate %d: found = false, want true", id)
		}
		cell, err := client.DecodeOperateValue(res.Values[0])
		if err != nil {
			t.Fatalf("DecodeOperateValue %d: %v", id, err)
		}
		if uint8(cell.U) != target { //nolint:gosec // U is a decoded U8 cell
			t.Fatalf("point %d: server rc = %d, want %d", id, cell.U, target)
		}
		finalRC[id] = target
	}

	f := vtypes.Filter{Op: vtypes.FilterGt, Field: "session/rc", Value: vtypes.NewInt(threshold)}
	pred, err := vector.CompileFilter(f)
	if err != nil {
		t.Fatalf("CompileFilter: %v", err)
	}

	want := make(map[uint64]bool, n)
	for id, rc := range finalRC {
		m := vtypes.Metadata{"session": vtypes.NewRecord(sessionRecordBytesRC(t, rc))}
		if pred(m) {
			want[id] = true
		}
	}
	if len(want) == 0 || len(want) == n {
		t.Fatalf("brute-force match set size %d — the fixture proves nothing (n=%d)", len(want), n)
	}

	resp, err := col.Search(ctx, client.SearchRequest{
		Query: []float32{0.1, 0.1, 0.1, 0.1}, K: n + 100, Filter: f,
	})
	if err != nil {
		t.Fatalf("Search: %v", err)
	}
	got := make(map[uint64]bool, len(resp.Results))
	for _, r := range resp.Results {
		got[r.ID] = true
	}

	for id := range want {
		if !got[id] {
			t.Errorf("Search is missing true match id %d (rc=%d)", id, finalRC[id])
		}
	}
	for id := range got {
		if !want[id] {
			t.Errorf("Search returned id %d (rc=%d) the predicate rejects", id, finalRC[id])
		}
	}
}
