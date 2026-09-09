// SPDX-License-Identifier: Apache-2.0

package vector

import (
	"errors"
	"fmt"
)

// RecordMutation is what a RecordMutator asks the engine to do with the record
// under the payload key. Three outcomes, not two: a failed CHECK must leave the
// point byte-identical AND unbumped, which "store these bytes" cannot express.
type RecordMutation uint8

const (
	RecordStore     RecordMutation = iota // store rec under the payload key
	RecordDelete                          // remove the payload key entirely
	RecordUnchanged                       // leave the point exactly as it was
)

// RecordMutator transforms the record bytes stored under one payload key.
//
// old aliases engine-owned memory and is valid ONLY for the duration of the
// call: a mutator must never retain it, never write through it, and never call
// back into the collection (it runs under the engine write lock). exists is
// false when the key is absent OR logically expired.
//
// OWNERSHIP HAND-OFF: rec goes the other way. The engine stores it BY REFERENCE
// (arena.SetMetadata: "The map is stored by reference; the caller must not
// mutate it after the hand-off"), so the mutator must return bytes it will never
// reuse, pool, or write through again. ops satisfies this because copyRecord
// allocates a fresh buffer per call and createRecord allocates fresh — NOT
// because the pooled operate engines promise it (schemaEngine.clear() nils
// e.buf rather than reusing it). A future pooled-buffer optimisation in ops
// would silently corrupt vector payloads; copyRecord carries a comment saying a
// caller now depends on its freshness.
type RecordMutator func(old []byte, exists bool) (rec []byte, act RecordMutation, err error)

// ErrPayloadKeyNotRecord is returned when the targeted payload key names
// something a record mutation cannot address: a key holding a non-record value,
// the empty key, or the reserved document-content key.
var ErrPayloadKeyNotRecord = errors.New("vector: payload key does not hold a record")

// ErrRecordMutation is returned when a RecordMutator reports an action outside
// the closed RecordMutation enumeration — a programming error in the caller,
// caught before any state change rather than defaulted to "store".
var ErrRecordMutation = errors.New("vector: record mutator returned an unknown mutation")

// ErrRecordMalformed is the ingest-side rejection for record bytes no operate
// engine can open. It is the shape half of the bound checkRecordValues already
// enforces on size, and it exists so a caller that hands the engine a torn
// record is told so at ingest instead of poisoning the payload index.
var ErrRecordMalformed = errors.New("vector: record payload value is not a decodable operate record")

// currentRecordValue reads the record bytes stored under key in meta, treating a
// key whose per-key deadline has passed as ABSENT (a logically expired key is
// invisible to every read path, so it must be invisible here too). It never
// copies: the returned slice aliases arena storage and is valid only under the
// engine lock.
//
// A key holding a non-record value is an ERROR, not an implicit overwrite. The
// caller can still read that value through get/search; silently replacing it
// with a record would destroy it on a call the caller believes only touches a
// record.
func currentRecordValue(meta Metadata, key string, expired bool) (rec []byte, exists bool, err error) {
	v, present := meta[key]
	if !present || expired {
		return nil, false, nil
	}
	if v.Kind != ValueRecord {
		return nil, false, fmt.Errorf("%w: payload key %q holds a value of kind %d", ErrPayloadKeyNotRecord, key, v.Kind)
	}
	return v.Rec, true, nil
}
