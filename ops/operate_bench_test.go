// SPDX-License-Identifier: Apache-2.0

package ops

// Benchmarks for the design doc's §2.6 in-place byte-patching claim: applying
// a small op list to an existing record should cost one allocation for the
// record copy, independent of how many rows the record holds. (The result frame
// used to be a second one; it is returned by value now.) Through handleOperate
// the record copy comes from a pool too — see BenchmarkOperateHandlerExistingRow. These measure that against the tree oracle, which rebuilds
// the whole record on every call, to quantify the win the byte engines exist
// for.
//
// benchBigSessionRecord and benchBigDynamicRecord mirror bigSessionRecord
// (operate_schema_engine_test.go) and bigDynamicRecord
// (operate_dynamic_engine_test.go) exactly, but take a *testing.B: the
// existing helpers are typed *testing.T and a benchmark cannot supply one.
// sessionArgsFor, dynamicArgsFor, applyTree, applyRecordBytes, and the path
// helpers (keyU64, fieldPath, colPath, opFromSchema) are reused unchanged —
// none of those need a *testing.T.

import (
	"bytes"
	"fmt"
	"testing"

	"github.com/rostamlabs/rostam/cache"
	"github.com/rostamlabs/rostam/sdk/wire"
)

func benchBigSessionRecord(b *testing.B, n int) []byte {
	b.Helper()
	s := sessionSchema()
	a := &wire.OperateArgs{Create: wire.OperateCreateSchema, Schema: s.Encode()}
	for k := 0; k < n; k++ {
		a.Ops = append(a.Ops, wire.OperateOp{Opcode: wire.OperateOpADD, Type: opFromSchema,
			Path: colPath(3, keyU64(uint64(k)), 0), A: 1})
	}
	out, _, _, err := applyRecordBytes(nil, a, 1)
	if err != nil {
		b.Fatal(err)
	}
	if _, derr := wire.DecodeRecord(out); derr != nil {
		b.Fatal(derr)
	}
	return out
}

func benchBigDynamicRecord(b *testing.B, n int) []byte {
	b.Helper()
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
		b.Fatal(err)
	}
	if _, derr := wire.DecodeRecord(out); derr != nil {
		b.Fatal(derr)
	}
	return out
}

// BenchmarkOperateSchemaExistingRow is the design doc's §2.6 headline number:
// the 6-op session op-list against an existing row in a 1024-row schema-mode
// record, through the in-place byte engine. applyRecordBytes returns a new
// buffer rather than mutating cur, so feeding it the same original bytes on
// every iteration keeps this on the in-place (found, no resize) path for the
// whole run.
func BenchmarkOperateSchemaExistingRow(b *testing.B) {
	rec := benchBigSessionRecord(b, 1024)
	a := sessionArgsFor(keyU64(512))
	a.Rets = nil
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, _, _, err := applyRecordBytes(rec, a, 1); err != nil {
			b.Fatal(err)
		}
	}
}

// BenchmarkOperateSchemaExistingRowWithReturns is the same call with the
// worked example's two Rets entries attached, informational: it shows the
// cost of populating a result frame on top of the in-place patch.
func BenchmarkOperateSchemaExistingRowWithReturns(b *testing.B) {
	rec := benchBigSessionRecord(b, 1024)
	key := keyU64(512)
	a := sessionArgsFor(key)
	a.Rets = []wire.OperateRet{
		{Mode: wire.OperateRetValue, Path: fieldPath(0)},
		{Mode: wire.OperateRetValue, Path: rowPath(3, key)},
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, _, _, err := applyRecordBytes(rec, a, 1); err != nil {
			b.Fatal(err)
		}
	}
}

// BenchmarkOperateSchemaNewRowEvict is the other end of §2.6: the table is
// already at its cap of 1024 rows, and every call targets a key the table
// does not hold, so every call inserts a row and evicts one to stay at cap.
// applyRecordBytes never mutates cur, so the base record — full at cap — is
// the same on every iteration; a single sentinel key that is never one of
// the base record's 1024 keys therefore takes the insert+evict path on every
// call without needing to build a fresh args value (and a fresh key) inside
// the timed loop.
func BenchmarkOperateSchemaNewRowEvict(b *testing.B) {
	rec := benchBigSessionRecord(b, 1024) // keys 0..1023, table cap 1024: full
	newKey := keyU64(1 << 32)             // never a key in the base record
	a := sessionArgsFor(newKey)
	a.Rets = nil
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, _, _, err := applyRecordBytes(rec, a, 1); err != nil {
			b.Fatal(err)
		}
	}
}

// BenchmarkOperateDynamicExistingRow is BenchmarkOperateSchemaExistingRow's
// dynamic-mode counterpart: the same 6-op shape (by name instead of by
// position) against an existing row in a 1024-row dynamic-mode record.
func BenchmarkOperateDynamicExistingRow(b *testing.B) {
	rec := benchBigDynamicRecord(b, 1024)
	a := dynamicArgsFor(keyU64(512))
	a.Rets = nil
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, _, _, err := applyRecordBytes(rec, a, 1); err != nil {
			b.Fatal(err)
		}
	}
}

// BenchmarkOperateTreeOracle runs the identical 6-op session call through the
// tree oracle instead of the byte engine: the comparison point for §2.6's
// claim that byte-patching avoids the oracle's per-call decode/rebuild/encode
// cost. applyTree copies its input before mutating it, so the decoded tree
// built once outside the loop stays valid input for every iteration.
func BenchmarkOperateTreeOracle(b *testing.B) {
	recBytes := benchBigSessionRecord(b, 1024)
	tree, err := wire.DecodeRecord(recBytes)
	if err != nil {
		b.Fatal(err)
	}
	a := sessionArgsFor(keyU64(512))
	a.Rets = nil
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, _, err := applyTree(tree, a, 1); err != nil {
			b.Fatal(err)
		}
	}
}

// BenchmarkOperateFreshRowInserts covers a call against a record that does not
// exist yet, with one op per key, so EVERY key opens a new row. That is the
// only path the regrow branches (insertGap for schema mode, splice for
// dynamic) are on - a call that only updates existing rows never opens a gap,
// and its cost is unchanged by growCap.
func BenchmarkOperateFreshRowInserts(b *testing.B) {
	for _, nRows := range []int{4, 16, 32} {
		b.Run(fmt.Sprintf("schema/rows=%d", nRows), func(b *testing.B) {
			s := sessionSchema()
			a := &wire.OperateArgs{Create: wire.OperateCreateSchema, Schema: s.Encode()}
			for k := 0; k < nRows; k++ {
				a.Ops = append(a.Ops, wire.OperateOp{Opcode: wire.OperateOpADD, Type: opFromSchema,
					Path: colPath(3, keyU64(uint64(k)), 0), A: 1})
			}
			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				if _, _, _, err := applyRecordBytes(nil, a, 1); err != nil {
					b.Fatal(err)
				}
			}
		})

		b.Run(fmt.Sprintf("dynamic/rows=%d", nRows), func(b *testing.B) {
			a := &wire.OperateArgs{Create: wire.OperateCreateDynamic}
			for k := 0; k < nRows; k++ {
				a.Ops = append(a.Ops, wire.OperateOp{Opcode: wire.OperateOpADD, Type: wire.OperateTypeU16,
					Path: nameColPath("b", keyU64(uint64(k)), "c"), A: 1})
			}
			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				if _, _, _, err := applyRecordBytes(nil, a, 1); err != nil {
					b.Fatal(err)
				}
			}
		})
	}
}

// BenchmarkOperateHandlerExistingRow drives a whole call through handleOperate,
// which is the only way the pooled buffers get exercised: every other benchmark
// here calls applyRecordBytes directly and so never touches them. What is left
// per call after the pools is the reply frame EncodeOperateResult builds.
func BenchmarkOperateHandlerExistingRow(b *testing.B) {
	cfg := cache.DefaultConfig()
	cfg.NumShards = 1
	c, err := cache.New(cfg)
	if err != nil {
		b.Fatal(err)
	}
	b.Cleanup(func() { _ = c.Close() })
	tx := NewTxContext(c)

	args, err := wire.EncodeOperateArgs(addField0([]byte("bench-handler-key")))
	if err != nil {
		b.Fatal(err)
	}
	if _, err := handleOperate(tx, args); err != nil { // create, and warm the pools
		b.Fatal(err)
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := handleOperate(tx, args); err != nil {
			b.Fatal(err)
		}
	}
}

// BenchmarkOperateAppendHandler and BenchmarkGetAppendHandler measure the
// AppendHandler twins against their allocating originals — the whole point of
// the append path being that a transport-owned buffer removes the reply
// allocation entirely.
func BenchmarkOperateAppendHandler(b *testing.B) {
	cfg := cache.DefaultConfig()
	cfg.NumShards = 1
	c, err := cache.New(cfg)
	if err != nil {
		b.Fatal(err)
	}
	b.Cleanup(func() { _ = c.Close() })
	tx := NewTxContext(c)
	args, err := wire.EncodeOperateArgs(addField0([]byte("bench-append-key")))
	if err != nil {
		b.Fatal(err)
	}
	if _, err := handleOperate(tx, args); err != nil {
		b.Fatal(err)
	}

	b.Run("Handler", func(b *testing.B) {
		b.ReportAllocs()
		for i := 0; i < b.N; i++ {
			if _, err := handleOperate(tx, args); err != nil {
				b.Fatal(err)
			}
		}
	})
	b.Run("AppendHandler", func(b *testing.B) {
		dst := make([]byte, 0, 512)
		b.ReportAllocs()
		for i := 0; i < b.N; i++ {
			out, err := handleOperateAppend(tx, args, dst[:0])
			if err != nil {
				b.Fatal(err)
			}
			dst = out
		}
	})
}

func BenchmarkGetAppendHandler(b *testing.B) {
	cfg := cache.DefaultConfig()
	cfg.NumShards = 1
	c, err := cache.New(cfg)
	if err != nil {
		b.Fatal(err)
	}
	b.Cleanup(func() { _ = c.Close() })
	tx := NewTxContext(c)
	key := []byte("bench-get-key")
	if err := tx.Put(key, bytes.Repeat([]byte("v"), 256), 0); err != nil {
		b.Fatal(err)
	}
	args := wire.EncodeKeyArgs(key)

	b.Run("Handler", func(b *testing.B) {
		b.ReportAllocs()
		for i := 0; i < b.N; i++ {
			if _, err := handleGet(tx, args); err != nil {
				b.Fatal(err)
			}
		}
	})
	b.Run("AppendHandler", func(b *testing.B) {
		dst := make([]byte, 0, 512)
		b.ReportAllocs()
		for i := 0; i < b.N; i++ {
			out, err := handleGetAppend(tx, args, dst[:0])
			if err != nil {
				b.Fatal(err)
			}
			dst = out
		}
	})
}
