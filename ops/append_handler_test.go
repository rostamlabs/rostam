// SPDX-License-Identifier: Apache-2.0

package ops

import (
	"bytes"
	"testing"

	"github.com/rostamlabs/rostam/sdk/wire"
)

// A registered AppendHandler must serve a read without allocating at all: the
// value goes straight into the caller's buffer instead of the fresh slice
// tx.Get returns. This is the whole point of the append path, and the budget
// fails if handleGetAppend ever starts allocating again (or if the op loses its
// variant and falls back).
func TestGetAppendHandlerIsZeroAlloc(t *testing.T) {
	if raceEnabled {
		t.Skip("sync.Pool.Put randomly drops items under -race by design, which defeats this allocation budget; see race_detect_test.go")
	}
	_, tx := newTestSetup(t)
	key := []byte("zero-alloc-key")
	val := bytes.Repeat([]byte("v"), 256)
	if err := tx.Put(key, val, 0); err != nil {
		t.Fatal(err)
	}
	args := wire.EncodeKeyArgs(key)
	dst := make([]byte, 0, 512)

	got := testing.AllocsPerRun(200, func() {
		out, err := handleGetAppend(tx, args, dst[:0])
		if err != nil {
			t.Fatal(err)
		}
		if !bytes.Equal(out, val) {
			t.Fatalf("got %d bytes, want %d", len(out), len(val))
		}
	})
	if got != 0 {
		t.Errorf("handleGetAppend allocated %.1f objects per read; want 0", got)
	}
}

// The append twin must agree with the original byte for byte — it is the same
// op, only the buffer differs. Each arm runs against its OWN key so both see
// identical store state: operate MUTATES, so running the two arms in sequence
// against one key would compare a create against an update and prove nothing.
func TestAppendHandlersMatchTheirHandlers(t *testing.T) {
	_, tx := newTestSetup(t)

	t.Run("get/hit", func(t *testing.T) {
		// A read does not mutate, so one key serves both arms.
		if err := tx.Put([]byte("k"), []byte("value"), 0); err != nil {
			t.Fatal(err)
		}
		args := wire.EncodeKeyArgs([]byte("k"))
		want, werr := handleGet(tx, args)
		if werr != nil {
			t.Fatal(werr)
		}
		dst := append(make([]byte, 0, 512), "PREFIX"...)
		got, gerr := handleGetAppend(tx, args, dst)
		if gerr != nil {
			t.Fatal(gerr)
		}
		if !bytes.HasPrefix(got, []byte("PREFIX")) {
			t.Fatalf("append overwrote the caller's existing bytes: %q", got)
		}
		if !bytes.Equal(got[len("PREFIX"):], want) {
			t.Errorf("payload mismatch:\n got %x\nwant %x", got[len("PREFIX"):], want)
		}
	})

	t.Run("operate/with returns", func(t *testing.T) {
		// Separate keys, identical ops, and a RET so the frame carries a value —
		// an empty frame would agree even if the value path were broken.
		mk := func(key []byte) *wire.OperateArgs {
			a := addField0(key)
			a.Rets = []wire.OperateRet{{Mode: wire.OperateRetValue, Path: fieldPath(0)}}
			return a
		}
		argsA, err := wire.EncodeOperateArgs(mk([]byte("agree-a")))
		if err != nil {
			t.Fatal(err)
		}
		argsB, err := wire.EncodeOperateArgs(mk([]byte("agree-b")))
		if err != nil {
			t.Fatal(err)
		}
		want, werr := handleOperate(tx, argsA)
		if werr != nil {
			t.Fatal(werr)
		}
		dst := append(make([]byte, 0, 512), "PREFIX"...)
		got, gerr := handleOperateAppend(tx, argsB, dst)
		if gerr != nil {
			t.Fatal(gerr)
		}
		if !bytes.HasPrefix(got, []byte("PREFIX")) {
			t.Fatalf("append overwrote the caller's existing bytes: %q", got)
		}
		frame := got[len("PREFIX"):]
		if len(frame) <= 3 {
			t.Fatalf("frame carries no return value (%d bytes); the arms would agree even if broken", len(frame))
		}
		if !bytes.Equal(frame, want) {
			t.Errorf("payload mismatch:\n got %x\nwant %x", frame, want)
		}
	})
}

// Registration wiring: the two hot ops must actually carry their variants, and
// an op without one must report so rather than erroring.
func TestAppendVariantsAreRegistered(t *testing.T) {
	r := NewRegistry()
	if err := RegisterBuiltins(r); err != nil {
		t.Fatal(err)
	}
	for _, name := range []string{"get", "operate"} {
		if _, ok := r.LookupAppend(name); !ok {
			t.Errorf("op %q has no append variant registered", name)
		}
	}
	if _, ok := r.LookupAppend("put"); ok {
		t.Error(`op "put" unexpectedly has an append variant`)
	}
	if _, ok := r.LookupAppend("nosuchop"); ok {
		t.Error("an unregistered op reported an append variant")
	}
}
