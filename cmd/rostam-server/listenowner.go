// SPDX-License-Identifier: Apache-2.0

package main

import (
	"bufio"
	"bytes"
	"context"
	"encoding/binary"
	"encoding/hex"
	"errors"
	"fmt"
	"io/fs"
	"net"
	"runtime"
	"strconv"
	"strings"
)

// portOwnerError means the address's listener belongs to another local user:
// it is not our daemon, and our token must not be sent to it.
type portOwnerError struct {
	port string
	uid  int
}

func (e *portOwnerError) Error() string {
	return fmt.Sprintf("port %s is owned by another user (uid %d); this plugin supports one user per host in this release", e.port, e.uid)
}

// listenerOwnerCheck verifies, before a dial to addr, that the process
// listening there runs as this user. It is a seam for tests; the real check is
// platform-specific (checkListenerOwner). It is a pre-check only: a listener
// can change between it and the dial, which connOwnerCheck closes.
//
// Both checks read the kernel's /proc socket table, so they run on Linux
// only. There they fail closed. Where the table does not exist at all (macOS,
// the BSDs) they return nil and the token is sent UNVERIFIED: the daemon is
// meant for single-user hosts.
var listenerOwnerCheck = checkListenerOwner

// connOwnerCheck verifies, after a dial and before anything is sent, who
// accepted the connection (see connOwnerErr). Seam for tests.
var connOwnerCheck = checkConnOwner

// ownerCheckBeforeConnect is listenerOwnerCheck as a client BeforeConnect
// hook, so the check runs before every dial, not only the first.
func ownerCheckBeforeConnect(_ context.Context, addr string) error {
	return listenerOwnerCheck(addr)
}

// ownerCheckAfterConnect is connOwnerCheck as a client AfterConnect hook: it
// runs after each dial and before the first request on that connection.
func ownerCheckAfterConnect(_ context.Context, addr string) error {
	return connOwnerCheck(addr)
}

// procListener is one LISTEN socket from /proc/net/tcp or /proc/net/tcp6.
type procListener struct {
	ip  net.IP
	v6  bool // from tcp6
	uid int
}

// procSocket is one line of /proc/net/tcp or /proc/net/tcp6.
type procSocket struct {
	localIP, remIP     net.IP
	localPort, remPort int
	state              string
	v6                 bool // from tcp6
	uid                int
}

const tcpListenState = "0A"

// procAddr decodes HEX_IP:HEX_PORT: the port big-endian, the IP printed as one
// (tcp) or four (tcp6) host-order 32-bit words.
func procAddr(s string) (net.IP, int, bool) {
	ipHex, portHex, ok := strings.Cut(s, ":")
	if !ok {
		return nil, 0, false
	}
	p, err := strconv.ParseUint(portHex, 16, 16)
	if err != nil {
		return nil, 0, false
	}
	raw, err := hex.DecodeString(ipHex)
	if err != nil || (len(raw) != net.IPv4len && len(raw) != net.IPv6len) {
		return nil, 0, false
	}
	ip := make(net.IP, len(raw))
	for i := 0; i < len(raw); i += 4 {
		binary.NativeEndian.PutUint32(ip[i:i+4], binary.BigEndian.Uint32(raw[i:i+4]))
	}
	return ip, int(p), true
}

// parseProcSockets returns every socket in the text of /proc/net/tcp or
// /proc/net/tcp6. Each line is "sl local rem st tx:rx tr:when retrnsmt uid
// ..."; lines that do not parse are skipped.
func parseProcSockets(text []byte) []procSocket {
	var out []procSocket
	sc := bufio.NewScanner(bytes.NewReader(text))
	for sc.Scan() {
		f := strings.Fields(sc.Text())
		if len(f) < 8 {
			continue
		}
		lip, lport, ok1 := procAddr(f[1])
		rip, rport, ok2 := procAddr(f[2])
		uid, err := strconv.Atoi(f[7])
		if !ok1 || !ok2 || err != nil || uid < 0 {
			continue
		}
		out = append(out, procSocket{localIP: lip, remIP: rip, localPort: lport, remPort: rport,
			state: f[3], v6: len(lip) == net.IPv6len, uid: uid})
	}
	return out
}

// parseProcNetTCP returns the LISTEN sockets on port from the text of
// /proc/net/tcp or /proc/net/tcp6.
func parseProcNetTCP(text []byte, port int) []procListener {
	var out []procListener
	for _, s := range parseProcSockets(text) {
		if s.state == tcpListenState && s.localPort == port {
			out = append(out, procListener{ip: s.localIP, v6: s.v6, uid: s.uid})
		}
	}
	return out
}

// procTableRequired is whether this platform has a /proc socket table. On
// Linux it must be readable: /proc unmounted or hidden (a sandbox) is an
// error, not a reason to skip the check. Elsewhere (macOS, the BSDs) there is
// none and the owner checks are skipped. A variable so tests can pick.
var procTableRequired = runtime.GOOS == "linux"

// readProcTables reads /proc/net/tcp and /proc/net/tcp6. ok is false only
// when neither exists on a platform without a /proc socket table (the caller
// then skips the check). Any read failure other than "does not exist", and
// both tables missing on Linux, are errors: Linux fails closed.
func readProcTables(read func(string) ([]byte, error)) (texts [][]byte, ok bool, err error) {
	for _, f := range []string{"/proc/net/tcp", "/proc/net/tcp6"} {
		b, rerr := read(f)
		switch {
		case rerr == nil:
			texts = append(texts, b)
		case errors.Is(rerr, fs.ErrNotExist):
		default:
			return nil, false, fmt.Errorf("cannot verify the listener's owner: reading %s: %w", f, rerr)
		}
	}
	if len(texts) == 0 && procTableRequired {
		return nil, false, errors.New("cannot verify the listener's owner: /proc/net/tcp and /proc/net/tcp6 are both missing (is /proc mounted?)")
	}
	return texts, len(texts) > 0, nil
}

// listenerOwnerErr is the decision behind checkListenerOwner: read is
// os.ReadFile, myUID os.Getuid(). It returns a *portOwnerError when the socket
// that would accept a connection to addr is owned by another uid, and nil when
// it is ours. It fails closed: an address that is not IP:port (a hostname
// could resolve anywhere, an empty host means every interface), a socket
// table that exists but cannot be read, and a readable table with no listener
// for addr are all errors. Only when there is no socket table at all (no
// /proc: macOS, the BSDs) does it return nil, unchecked.
func listenerOwnerErr(addr string, read func(string) ([]byte, error), myUID int) error {
	host, portStr, err := net.SplitHostPort(addr)
	if err != nil {
		return fmt.Errorf("cannot verify the owner of %q: %w", addr, err)
	}
	target := net.ParseIP(host)
	if target == nil {
		return fmt.Errorf("cannot verify the owner of %s: host %q is not an IP address", addr, host)
	}
	port, err := strconv.Atoi(portStr)
	if err != nil || port < 1 || port > 65535 {
		return fmt.Errorf("cannot verify the owner of %s: bad port %q", addr, portStr)
	}
	texts, ok, err := readProcTables(read)
	if err != nil {
		return err
	}
	if !ok {
		return nil // no socket table on this platform: best effort
	}
	var all []procListener
	for _, b := range texts {
		all = append(all, parseProcNetTCP(b, port)...)
	}
	// A socket bound to the exact address takes the connection over a
	// wildcard one, so wildcards only decide when there is no exact match.
	var exact, wild []procListener
	for _, l := range all {
		switch {
		case l.ip.Equal(target): // also matches ::ffff:a.b.c.d for a v4 target
			exact = append(exact, l)
		case l.ip.IsUnspecified() && (l.v6 || target.To4() != nil):
			wild = append(wild, l)
		}
	}
	candidates := exact
	if len(candidates) == 0 {
		candidates = wild
	}
	if len(candidates) == 0 {
		// Nothing listens there right now, so nothing can be verified; a
		// listener that appears after this check is exactly what must not get
		// the token.
		return fmt.Errorf("cannot verify the owner of %s: no listening socket found", addr)
	}
	for _, l := range candidates {
		if l.uid != myUID {
			return &portOwnerError{port: portStr, uid: l.uid}
		}
	}
	return nil
}

// connOwnerErr runs after a dial to addr and before the first request. The
// pinned client module does not expose the new connection's own endpoints, so
// it checks every connection this uid has ESTABLISHED to addr (the new one is
// always among them): the server end of each (local = addr, remote = our
// end), which the kernel creates with the listening socket's uid, must be
// ours. One owned by another uid is a *portOwnerError; a server end that
// cannot be found, or no connection to addr at all, cannot be verified and is
// an error. This closes the window between listenerOwnerErr and the dial:
// whoever accepted the connection is what is checked. It is conservative: a
// connection of ANY of this user's processes to addr whose peer is foreign
// refuses this one too, which is the right answer for a hijacked port.
//
// Same platform rules as listenerOwnerErr (see readProcTables).
func connOwnerErr(addr string, read func(string) ([]byte, error), myUID int) error {
	host, portStr, err := net.SplitHostPort(addr)
	if err != nil {
		return fmt.Errorf("cannot verify the owner of %q: %w", addr, err)
	}
	target := net.ParseIP(host)
	port, perr := strconv.Atoi(portStr)
	if target == nil || perr != nil {
		return fmt.Errorf("cannot verify the owner of %s: not an IP:port", addr)
	}
	texts, ok, err := readProcTables(read)
	if err != nil {
		return err
	}
	if !ok {
		return nil // no socket table on this platform: unchecked
	}
	var all []procSocket
	for _, b := range texts {
		all = append(all, parseProcSockets(b)...)
	}
	const established = "01"
	ours := 0
	for _, c := range all {
		if c.state != established || c.uid != myUID || c.remPort != port || !c.remIP.Equal(target) {
			continue
		}
		ours++
		found := false
		for _, s := range all {
			if s.state == tcpListenState || s.localPort != port || s.remPort != c.localPort ||
				!s.localIP.Equal(target) || !s.remIP.Equal(c.localIP) {
				continue
			}
			found = true
			if s.uid != myUID {
				return &portOwnerError{port: portStr, uid: s.uid}
			}
		}
		if !found {
			return fmt.Errorf("cannot verify the owner of %s: the server end of a connection to it is not in the socket table", addr)
		}
	}
	if ours == 0 {
		return fmt.Errorf("cannot verify the owner of %s: no established connection to it found", addr)
	}
	return nil
}
