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

// recordKeyPastDeadline reports whether the record under a mutated payload key
// must be treated as ABSENT because its per-key deadline has passed — and it
// answers false, always, on an UNSTAMPED apply.
//
// THE ASYMMETRY IS THE WHOLE POINT, AND IT IS NOT THE SAME AS set_payload's.
// A replicated apply runs the op-list inside every replica's own FSM. With the
// leader apply stamp off (its rollout flag is off cluster-wide today) each
// replica computes `now` from its OWN wall clock. If the mutator were allowed to
// judge the key's deadline against that clock, a replica whose clock is past the
// deadline would read the record as absent and CREATE A FRESH ONE, while a
// replica whose clock is not yet past it would increment the existing one. Both
// commit, both bump the version, neither errors, and the two replicas hold
// permanently different record BYTES.
//
// setPayloadBody (vector/hnsw.go) has no such class: it copies an expired key
// through unchanged and only ever moves a DEADLINE value, so its wall-clock read
// costs a bounded millisecond skew in a deadline, never divergent stored bytes.
// A record mutation reads the stored bytes and writes a function of them, so the
// same wall-clock read converts that bounded skew into unbounded content
// divergence. Hence: only the STAMPED branch — whose clock is the leader's
// stamp, identical on every replica and therefore deterministic — may expire the
// key. The unstamped branch treats the record as PRESENT and passes its deadline
// through untouched, which is exactly what setPayloadBody does with an expired
// key it is not asked to re-deadline.
//
// The consequence, stated plainly so it is not rediscovered as a bug: an
// unstamped vector_operate against a key whose deadline has passed mutates the
// record in place and keeps the stale deadline, rather than starting a new
// record. The key is still invisible to every READ path (which judges liveness
// at read time), so the only observable difference is what the next read sees
// after the deadline is refreshed. Determinism is worth that.
//
// The POINT-level liveness gate is deliberately NOT covered by this rule: it is
// judged against the wall clock on the unstamped branch exactly as
// setPayloadBody's is, because that is a pre-existing, phase-wide property of
// every unstamped write and changing it here would make record mutation
// disagree with its siblings about whether a point exists.
func recordKeyPastDeadline(stamped bool, deadline, now uint64) bool {
	return stamped && keyExpired(deadline, now)
}

// recordKeyPastDeadlineAbs is recordKeyPastDeadline for the named/multi-vector
// families, whose per-key deadlines are ABSOLUTE unix-millis int64 and which
// have no keyExpired twin (liveMetaMap's rule, inlined). Same stamped-only
// contract, same reasons — see recordKeyPastDeadline.
func recordKeyPastDeadlineAbs(stamped bool, deadline, now int64) bool {
	return stamped && deadline != 0 && deadline <= now
}

// currentRecordValue reads the record bytes stored under key in meta, treating a
// key whose per-key deadline has passed as ABSENT (a logically expired key is
// invisible to every read path, so it must be invisible here too). It never
// copies: the returned slice aliases arena storage and is valid only under the
// engine lock.
//
// WHO DECIDES `expired`: the four mutate bodies, through
// recordKeyPastDeadline[Abs], which answer false on an UNSTAMPED apply so a
// replica's wall clock can never decide whether a record exists. Read that
// helper's doc before changing anything here — this function's "expired means
// absent" rule is only safe because the caller refuses to evaluate it against a
// per-replica clock.
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
