// SPDX-License-Identifier: Apache-2.0

package ops

import (
	"errors"
	"fmt"
	"strconv"
	"strings"
)

// ErrOperateDuringReshard is the refusal the root store raises when a
// vector_operate targets a collection a reshard is currently dual-writing. It is
// RETRYABLE: the caller retries after cutover, when the collection has a single
// live generation again. rostam.ErrOperateDuringReshard is this same value — see
// there for WHY operate refuses where set_payload dual-writes.
//
// WHY IT IS DECLARED HERE and not beside the code that raises it. The refusal is
// only useful if the caller is told to retry, and that means every transport
// classifier has to recognise it. server, httpapi and grpcapi cannot import the
// root package without a layering cycle — which is why the alias and key-admin
// errors in those classifiers are matched by message prefix. Declaring the
// sentinel in a package all four already import lets them match by IDENTITY,
// with the message shape only as the clustered-stringification fallback, so a
// rewording is a compile-time concern rather than a silent regression to
// "internal error". Same reason WASMUpdateUnsupportedMsg is a const here.
//
// The message keeps the "rostam: " prefix because the ROOT store is what
// refuses; the text is the caller-visible contract docs/vector/filtering.md
// quotes.
var ErrOperateDuringReshard = errors.New("rostam: vector_operate is refused while a reshard is dual-writing; retry after cutover")

// OperateDuringReshardErr is the ONE producer of the refusal's detailed form,
// naming the collection so an operator reading a log knows which reshard to wait
// on. The collection name goes through clipOperateName, not %q: it is
// caller-supplied and bounded only by the route body cap at this point (the
// 255-byte wire cap is applied later, by EncodeVectorOperateArgs), and this
// message is returned VERBATIM to the caller on every transport.
func OperateDuringReshardErr(collection string) error {
	return fmt.Errorf("%w: collection %s", ErrOperateDuringReshard, clipOperateName(collection))
}

// clipOperateName renders a caller-supplied name for an error message: the whole
// thing quoted when it is short, otherwise a 64-byte quoted prefix plus the full
// length. It is vector.clipField's rule, restated here because that one is
// unexported and this package must not widen vector's API to borrow it. Keep the
// two in step: the bound is what stops a caller-chosen name from setting the
// size of an error message, and of the classification scan that reads it.
func clipOperateName(s string) string {
	if len(s) <= maxClippedNameBytes {
		return strconv.Quote(s)
	}
	return fmt.Sprintf("%s… (%d bytes)", strconv.Quote(s[:maxClippedNameBytes]), len(s))
}

// maxClippedNameBytes is the number of SOURCE bytes clipOperateName shows
// before it truncates. It is shared with isClippedName so the matcher enforces
// the producer's own bound rather than a second copy of the number.
const maxClippedNameBytes = 64

// IsOperateDuringReshardMessage reports whether s is the EXACT serialised form
// of an ErrOperateDuringReshard error — anchored the way
// vector.IsRecordTooLargeMessage and IsVectorRecordAbsentMessage are for their
// sentinels, and for the same reason: shard.decodePBResult rebuilds a replicated
// op error with errors.New, losing errors.Is identity, and a bare
// strings.Contains fallback would make any error that merely mentions the
// sentinel text client-facing.
//
// The two recognized shapes (<name> is clipOperateName's bounded rendering of
// the caller's collection name):
//
//	rostam: vector_operate is refused while a reshard is dual-writing; retry after cutover
//	rostam: vector_operate is refused while a reshard is dual-writing; retry after cutover: collection <name>
//
// The bare form exists only for a caller that stringifies the sentinel itself;
// every production site goes through OperateDuringReshardErr. The far end is
// anchored as tightly as the clipped rendering allows: what follows the fixed
// prefix must LOOK like clipOperateName's output — a quoted string, optionally
// followed by the "… (N bytes)" length tail — so an internal fault that appends
// its own trailing context ("... : collection \"c\" (shard 3 apply failed)") is
// declined rather than leaked.
func IsOperateDuringReshardMessage(s string) bool {
	if len(s) == 0 || len(s) > maxOperateDuringReshardMessageLen {
		return false
	}
	if s == ErrOperateDuringReshard.Error() {
		return true
	}
	rest, ok := strings.CutPrefix(s, operateDuringReshardPrefix)
	if !ok {
		return false
	}
	return isClippedName(rest)
}

// maxOperateDuringReshardMessageLen bounds the input
// IsOperateDuringReshardMessage scans, so classification cost cannot scale with
// an attacker-chosen collection name. clipOperateName already bounds the name to
// 64 source bytes (worst-case quoted/escaped well under 300), so every real
// message sits far below this; mirrors maxRecordTooLargeMessageLen's reasoning.
const maxOperateDuringReshardMessageLen = 512

// operateDuringReshardPrefix is the fixed text that opens the detailed form,
// ending right before the caller-controlled clipOperateName rendering.
var operateDuringReshardPrefix = ErrOperateDuringReshard.Error() + ": collection "

// clippedNameLengthTail is the fixed text that closes clipOperateName's
// truncated form, after the decimal byte count.
const clippedNameLengthTail = " bytes)"

// isClippedName reports whether s is exactly what clipOperateName renders, and
// enforces the producer's own INVARIANTS rather than merely its punctuation:
//
//   - untruncated form: a canonical Go-quoted string whose DECODED length is at
//     most maxClippedNameBytes. clipOperateName only takes this branch for a
//     name that short, so a longer one is a shape it can never emit.
//   - truncated form: a canonical Go-quoted prefix of EXACTLY
//     maxClippedNameBytes decoded bytes, then "… (N bytes)" with N strictly
//     greater than maxClippedNameBytes — the branch is only taken when the full
//     name is longer than the prefix it shows.
//
// Without the two length rules the matcher accepted forms the producer cannot
// emit (a 65-byte quoted name with no length tail, a truncated form whose prefix
// is 63 bytes), which is how an unrelated internal error carrying the fixed
// refusal prefix could be classified as the retryable refusal and cross the
// wire verbatim instead of being redacted.
func isClippedName(s string) bool {
	if rest, ok := strings.CutSuffix(s, clippedNameLengthTail); ok {
		i := len(rest)
		for i > 0 && rest[i-1] >= '0' && rest[i-1] <= '9' {
			i--
		}
		if i == len(rest) {
			return false // no digits where the byte count belongs
		}
		// The count is the FULL source length, which the truncated branch is
		// only reached for when it exceeds the shown prefix. ParseUint (not
		// Atoi) so the width does not depend on GOARCH; an overflowing run of
		// digits is a shape clipOperateName cannot produce either.
		n, perr := strconv.ParseUint(rest[i:], 10, 64)
		if perr != nil || n <= maxClippedNameBytes {
			return false
		}
		quoted, ok := strings.CutSuffix(rest[:i], "… (")
		if !ok {
			return false
		}
		dec, ok := goQuoted(quoted)
		return ok && len(dec) == maxClippedNameBytes
	}
	dec, ok := goQuoted(s)
	return ok && len(dec) <= maxClippedNameBytes
}

// goQuoted reports whether s is a CANONICAL strconv.Quote rendering, and
// returns what it decodes to so the caller can bound the decoded length.
//
// Unquote alone is not enough. It accepts renderings strconv.Quote never
// produces — "\x41" for "A", say, or an escaped character Quote would leave
// literal — and a matcher that accepted those would be accepting text no
// producer in this tree can emit. Re-quoting the decoded string and requiring
// byte equality is the whole canonicality check, and it subsumes Unquote's own
// rejections (an unterminated quote, a stray byte after the closing quote, an
// invalid escape).
func goQuoted(s string) (string, bool) {
	if len(s) < 2 || s[0] != '"' {
		return "", false
	}
	dec, err := strconv.Unquote(s)
	if err != nil {
		return "", false
	}
	if strconv.Quote(dec) != s {
		return "", false
	}
	return dec, true
}
