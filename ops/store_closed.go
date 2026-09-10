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

// maxStoreClosedMessageLen bounds the input IsStoreClosedMessage scans, so
// classification cost cannot scale with an attacker-influenced error string. The
// real message is the sentinel plus at most the two wrappers below; mirrors
// maxKVQueryNoSuchIndexMessageLen's reasoning.
const maxStoreClosedMessageLen = 256

// maxStoreClosedWrappers bounds how many wrappers are peeled before giving up.
// Two are reachable in production (the fan-out's, and the peer client's nested
// inside it); the limit is what stops a crafted run of repeated prefixes from
// setting the cost of one classification.
const maxStoreClosedWrappers = 4

// The two wrapper prefixes a store-closed refusal is known to arrive behind,
// split at the point where each becomes variable:
//
//	cluster: kv_query: shard group <n>: <inner>
//	client: server error on op "<op>": <inner>
//
// The first is broadcastKVQuery wrapping one group's error; the second is
// client.RemoteError rendering a peer's reply. Both are ANCHORED CUTS, never
// searches.
const (
	kvFanoutWrapperOpen  = "cluster: kv_query: shard group "
	kvRemoteWrapperOpen  = `client: server error on op "`
	kvRemoteWrapperClose = `": `
	kvWrapperSep         = ": "
)

// IsStoreClosedMessage reports whether s IS shard.ErrStoreClosed — bare, or
// behind the wrappers the fan-out and the peer client put in front of it — for
// a transport that cannot match it by identity (httpapi and grpcapi cannot
// import shard).
//
// WHY EXACT FORM AND NOT A SUFFIX TEST. This predicate makes an error
// RETRYABLE, and a caller told to retry does — so anything that can be made to
// LOOK like this message is a way to make a permanent error retry forever, at
// one full cluster fan-out per attempt. An earlier version of this function
// matched a SUFFIX with a single veto on the filter sentinel, and a suffix is
// satisfiable by any caller-controlled text that lands at the END of some other
// error: a filter refusal quotes the caller's own field name verbatim, and an
// index name is quoted into the no-such-index refusal, so either could be shaped
// to end in the sentinel, and the veto only covered one of the two.
//
// The rule instead is the one IsKVQueryNoSuchIndexMessage follows: peel only the
// KNOWN wrappers, each by an anchored cut whose variable part must have the
// shape that wrapper actually produces, then require what REMAINS to EQUAL the
// sentinel. No caller text survives that, because caller text is never the whole
// of what remains — the sentinel that produced the message is always in front of
// it, and that prefix is not a wrapper this function will peel.
//
// The veto is gone with the suffix it guarded: exactness subsumes it, and a dead
// check with a live-sounding comment is how the next reader is misled.
func IsStoreClosedMessage(s string) bool {
	if len(s) == 0 || len(s) > maxStoreClosedMessageLen {
		return false
	}
	for i := 0; i <= maxStoreClosedWrappers; i++ {
		if s == StoreClosedMsg {
			return true
		}
		rest, ok := cutStoreClosedWrapper(s)
		if !ok {
			return false
		}
		s = rest
	}
	return false
}

// cutStoreClosedWrapper removes ONE known wrapper from the front of s, or
// reports false when s does not open with one. Each cut is anchored at position
// zero, and each wrapper's variable part must have the shape its producer emits
// — so a message that merely resembles a wrapper is declined rather than peeled
// into something that then compares equal.
func cutStoreClosedWrapper(s string) (string, bool) {
	if rest, ok := strings.CutPrefix(s, kvFanoutWrapperOpen); ok {
		// "<shard group ordinal>: <inner>": digits, then the separator.
		i := 0
		for i < len(rest) && rest[i] >= '0' && rest[i] <= '9' {
			i++
		}
		if i == 0 {
			return "", false // nothing where the group ordinal belongs
		}
		return strings.CutPrefix(rest[i:], kvWrapperSep)
	}
	if rest, ok := strings.CutPrefix(s, kvRemoteWrapperOpen); ok {
		// `<op name>": <inner>`. The op name is rendered with %q from one of
		// this process's own constants, never from caller input, so it must LOOK
		// like an op name rather than merely be terminated by a quote.
		j := strings.Index(rest, kvRemoteWrapperClose)
		if j < 1 || !looksLikeOpName(rest[:j]) {
			return "", false
		}
		return rest[j+len(kvRemoteWrapperClose):], true
	}
	return "", false
}

// looksLikeOpName reports whether s is in the charset every op name in this
// system uses. An op name reaching a wrapper is one of this process's own
// constants, so anything outside the charset means that wrapper did not write
// this text — which is what an arbitrary quoted string looks like, since %q
// escapes anything unusual.
func looksLikeOpName(s string) bool {
	if len(s) == 0 || len(s) > 64 {
		return false
	}
	for i := 0; i < len(s); i++ {
		c := s[i]
		switch {
		case c >= 'a' && c <= 'z', c >= 'A' && c <= 'Z', c >= '0' && c <= '9', c == '_':
		default:
			return false
		}
	}
	return true
}
