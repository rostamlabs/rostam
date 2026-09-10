// SPDX-License-Identifier: Apache-2.0

package ops

import "strings"

// StoreClosedMsg is the text of shard.ErrStoreClosed — the refusal a shard
// store returns once Close has begun and it will not admit another Call.
//
// WHY IT IS DECLARED HERE and not only beside the error. The refusal is only
// useful if the caller is told to retry against another replica, and that means
// every transport classifier has to recognise it. httpapi and grpcapi cannot
// import shard (see server.clientFacingErr's note on the same wall), so without
// a shared spelling they would each carry their own copy of the string and
// discover a rewording in production. shard declares the sentinel AS
// errors.New(ops.StoreClosedMsg), so the two can never drift — same reason
// WASMUpdateUnsupportedMsg and ErrOperateDuringReshard live in this package.
const StoreClosedMsg = "shard: store is closed"

// storeClosedWrapped is the tail a wrapped refusal ends with. The message
// carries NO caller-supplied text of its own, so the whole classification is
// "the error ends where this refusal ends".
const storeClosedWrapped = ": " + StoreClosedMsg

// maxStoreClosedMessageLen bounds the input IsStoreClosedMessage scans, so
// classification cost cannot scale with an attacker-influenced error string. The
// real message is the sentinel plus at most a couple of wrapper prefixes (a
// shard-group ordinal, a peer op name), all far below this; mirrors
// maxOperateDuringReshardMessageLen's reasoning.
const maxStoreClosedMessageLen = 512

// IsStoreClosedMessage reports whether s is shard.ErrStoreClosed as it reaches a
// transport that cannot match it by identity: bare, or as the tail of the
// wrappers the fan-out and the peer client put in front of it
// ("cluster: kv_query: shard group 3: …", "client: server error on op %q: …").
//
// WHY A SUFFIX AND A VETO, NOT strings.Contains. This predicate makes an error
// RETRYABLE, and a caller told to retry does — so any string a client can get
// into an error message is, under a substring test, a way to make a permanent
// error retry forever. The filter refusal quotes the caller's own field name
// verbatim, so a field named after this sentinel defeats a substring check
// exactly. Two things close it: a message carrying the permanent filter
// sentinel's text is never reclassified, whatever else it says; and what remains
// must END with the refusal, which no quoted field name can (%q leaves a closing
// quote behind it).
//
// Contains is used for the veto and ONLY for the veto, where it can only refuse
// a rewrite — a false positive there costs a retryable error reported as
// permanent, which is the safe direction. Same shape cluster.isKVQueryNoSuchIndex
// uses.
func IsStoreClosedMessage(s string) bool {
	if len(s) == 0 || len(s) > maxStoreClosedMessageLen {
		return false
	}
	if s == StoreClosedMsg {
		return true
	}
	if strings.Contains(s, ErrKVQueryFilter.Error()) {
		return false
	}
	return strings.HasSuffix(s, storeClosedWrapped)
}
