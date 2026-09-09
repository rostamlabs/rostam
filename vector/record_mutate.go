// SPDX-License-Identifier: Apache-2.0

package vector

import (
	"errors"
	"fmt"
	"strings"
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
		// The key goes through clipField, not %q: it is caller-supplied and
		// unbounded, and this message is returned VERBATIM to the caller on
		// every transport (a client error, classified 400 / InvalidArgument)
		// and carried across replication as a string. Keep in step with
		// checkRecordValues / checkRecordValuesAll (vector/metadata.go), which
		// clip the key for the same reason.
		return nil, false, fmt.Errorf("%w: payload key %s holds a value of kind %d", ErrPayloadKeyNotRecord, clipField(key), v.Kind)
	}
	return v.Rec, true, nil
}

// IsPayloadKeyNotRecordMessage reports whether s is the EXACT serialised form
// of an ErrPayloadKeyNotRecord error — anchored the same way
// IsRecordTooLargeMessage (vector/metadata.go) is for its sentinel, and for
// the same reason: shard.decodePBResult rebuilds a replicated op error with
// errors.New, losing errors.Is identity, and a bare strings.Contains fallback
// would make any error that merely mentions the sentinel text client-facing.
//
// The two recognized shapes (<key> is clipField's bounded rendering of the
// caller's payload key, exactly as for ErrRecordMalformed/ErrRecordTooLarge;
// <N> is the decimal vtypes.ValueKind of whatever the key actually holds):
//
//	vector: payload key does not hold a record
//	vector: payload key does not hold a record: payload key <key> holds a value of kind <N>
//
// The bare form comes from mutatePayloadRecordLockedAt / mutatePayloadRecordBody's
// own guard clause (fn == nil, key == "", or key == contentField); the detailed
// form comes from currentRecordValue, for a key that resolves to a non-record
// value. Neither ever passes through a bulk "payload N: " wrapper — a
// vector_operate always targets exactly one payload key on exactly one point,
// unlike the ingest paths ErrRecordMalformed/ErrRecordTooLarge share.
//
// Like IsRecordTooLargeMessage's numeric tail, the kind digits sit at the very
// end of the string with nothing after them, so — unlike
// IsRecordMalformedMessage — the far end IS fully anchored: this reports false
// for anything that does not end in "... holds a value of kind " followed by
// one or more digits and nothing else.
func IsPayloadKeyNotRecordMessage(s string) bool {
	if len(s) == 0 || len(s) > maxPayloadKeyNotRecordMessageLen {
		return false
	}
	if s == ErrPayloadKeyNotRecord.Error() {
		return true
	}
	rest, ok := strings.CutPrefix(s, payloadKeyNotRecordPrefix)
	if !ok {
		return false
	}
	return hasPayloadKeyNotRecordSuffix(rest)
}

// maxPayloadKeyNotRecordMessageLen bounds the input IsPayloadKeyNotRecordMessage
// scans, so classification cost cannot scale with an attacker-chosen key's
// length. clipField already bounds the key to ~64 source bytes (worst-case
// quoted/escaped well under 300), so — unlike before clipField was used here —
// even a caller-chosen key of arbitrary length (a 1 MiB key, say) produces a
// message well under this cap; mirrors maxRecordTooLargeMessageLen's reasoning.
const maxPayloadKeyNotRecordMessageLen = 512

// payloadKeyNotRecordPrefix is the fixed text that opens the detailed form,
// ending right before the caller-controlled clipField(key) rendering.
var payloadKeyNotRecordPrefix = ErrPayloadKeyNotRecord.Error() + ": payload key "

// payloadKeyNotRecordSuffixMid is the fixed text between the caller-controlled
// key rendering and the trailing kind digits.
const payloadKeyNotRecordSuffixMid = " holds a value of kind "

// hasPayloadKeyNotRecordSuffix reports whether s ends with the fixed marker
// text followed by one or more decimal digits and nothing else — the same
// digit-stripping-then-anchor approach hasRecordTooLargeSuffix uses.
func hasPayloadKeyNotRecordSuffix(s string) bool {
	i := len(s)
	for i > 0 && s[i-1] >= '0' && s[i-1] <= '9' {
		i--
	}
	if i == len(s) {
		return false
	}
	return strings.HasSuffix(s[:i], payloadKeyNotRecordSuffixMid)
}
