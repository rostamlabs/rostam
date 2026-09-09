// SPDX-License-Identifier: Apache-2.0

package vector

import (
	"bytes"
	"testing"

	"github.com/rostamlabs/rostam/sdk/wire"
)

// A record payload follows the SAME ownership policy as the three list kinds:
// the slice is held by reference in both directions, and the engine's metadata
// copies are map-level (cloneMeta, and the inline map copy in Get). So an
// IN-PROCESS reader is handed Values whose slices still point at stored bytes,
// for records and for strings/ints/floats alike, and writing through one is a
// caller contract violation — vtypes.Metadata documents it for all four kinds.
//
// This test pins the two halves of that policy that actually matter:
//
//  1. The WIRE path hands out nothing of the engine's. Every network client
//     reads a payload the codec wrote into its own buffer, so mutating it
//     cannot reach stored bytes. That is what makes the in-process aliasing a
//     local, documented contract rather than a remotely reachable hazard.
//  2. Records are not a NEW hazard: a string list behaves identically, which is
//     why the fix for one would have to be the fix for all four.
func TestGetPayloadWirePathIsCallerOwned(t *testing.T) {
	const dim = 4
	h, err := newHNSW(Config{Dim: dim, Metric: L2, M: 16, EfConstruction: 100, EfSearch: 32, Seed: 1})
	if err != nil {
		t.Fatal(err)
	}
	rec := sessionRecordBytes(t)
	stored := append([]byte(nil), rec...) // the reference copy, never handed to the engine
	meta := Metadata{
		"session": NewRecord(append([]byte(nil), rec...)),
		"tags":    NewStrings([]string{"alpha", "beta"}),
	}
	vec := []float32{1, 2, 3, 4}
	if _, _, err := h.Insert(1, vec, 0, meta, nil, nil, CASCond{}); err != nil {
		t.Fatal(err)
	}

	// Read the point the way a network client does: through the result codec.
	gv, gm, ttl, sparse, version, ok := h.Get(1)
	if !ok {
		t.Fatal("Get(1) = !ok")
	}
	body := wire.EncodeVectorGetResultV(true, gv, gm, ttl, sparse, true, true, version)
	_, _, wm, _, _, _, err := wire.DecodeVectorGetResultV(body)
	if err != nil {
		t.Fatalf("DecodeVectorGetResultV: %v", err)
	}

	// Mutate everything the decoded payload holds, as hostilely as a caller can.
	wrec := wm["session"]
	if wrec.Kind != ValueRecord || !bytes.Equal(wrec.Rec, stored) {
		t.Fatalf("decoded record = %v, want the stored bytes", wrec)
	}
	for i := range wrec.Rec {
		wrec.Rec[i] ^= 0xFF
	}
	wtags := wm["tags"]
	if wtags.Kind != ValueStrings || len(wtags.Strs) != 2 {
		t.Fatalf("decoded tags = %v, want two strings", wtags)
	}
	wtags.Strs[0] = "MUTATED"

	// The engine is untouched: the stored record still resolves, the synthetic
	// postings still describe it, and the list value is unchanged.
	_, after, _, _, _, ok := h.Get(1)
	if !ok {
		t.Fatal("Get(1) after the mutation = !ok")
	}
	if got := after["session"]; !bytes.Equal(got.Rec, stored) {
		t.Errorf("stored record changed through the wire payload:\n got %x\nwant %x", got.Rec, stored)
	}
	if got := after["tags"]; got.Strs[0] != "alpha" {
		t.Errorf("stored string list changed through the wire payload: %q", got.Strs)
	}
	// The predicate is the real consumer of those bytes: a corrupted record
	// would stop resolving, so this is the assertion that the STORED record is
	// still a record and still says what it said.
	pred := compileOrFail(t, Filter{Op: FilterEq, Field: "session/rc", Value: NewInt(7)})
	if !pred(after) {
		t.Error("session/rc == 7 no longer matches the stored record")
	}
	rowPred := compileOrFail(t, Filter{Op: FilterRowExists, Field: "session/b/42"})
	if !rowPred(after) {
		t.Error("row_exists(session/b/42) no longer matches the stored record")
	}
}

// TestGetPayloadInProcessAliasingIsUniform is the companion fact, asserted so
// the policy is visible rather than inferred: an in-process Get hands back
// Values that ALIAS stored bytes, and it does so identically for a record and
// for a string list. If a future change deep-copies one of them on the read
// path it must deep-copy all four kinds, and this test will say so.
func TestGetPayloadInProcessAliasingIsUniform(t *testing.T) {
	const dim = 4
	h, err := newHNSW(Config{Dim: dim, Metric: L2, M: 16, EfConstruction: 100, EfSearch: 32, Seed: 1})
	if err != nil {
		t.Fatal(err)
	}
	meta := Metadata{
		"session": NewRecord(sessionRecordBytes(t)),
		"tags":    NewStrings([]string{"alpha"}),
	}
	if _, _, err := h.Insert(1, []float32{1, 2, 3, 4}, 0, meta, nil, nil, CASCond{}); err != nil {
		t.Fatal(err)
	}
	_, a, _, _, _, ok := h.Get(1)
	if !ok {
		t.Fatal("Get(1) = !ok")
	}
	_, b, _, _, _, ok := h.Get(1)
	if !ok {
		t.Fatal("second Get(1) = !ok")
	}
	// Two independent reads share the same backing arrays: the copies are
	// map-level, so the slice headers still point at one set of bytes.
	if &a["session"].Rec[0] != &b["session"].Rec[0] {
		t.Error("record payload was copied on the read path; the list kinds are not — make the policy uniform")
	}
	if &a["tags"].Strs[0] != &b["tags"].Strs[0] {
		t.Error("string-list payload was copied on the read path; the record kind is not — make the policy uniform")
	}
}
