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

// connOwnerCheck verifies an established connection's peer before anything is
// sent on it (see connOwnerErr). Seam for tests, like listenerOwnerCheck.
var connOwnerCheck = checkConnOwner

// ownerCheckBeforeConnect is listenerOwnerCheck as a client BeforeConnect
// hook, so the check runs before every dial, not only the first.
func ownerCheckBeforeConnect(_ context.Context, addr string) error {
	return listenerOwnerCheck(addr)
}

// ownerCheckVerifyConn is connOwnerCheck as a client VerifyConn hook: it runs
// on the connection the token would travel over, after the dial and before
// the first request, so there is no window between check and use.
func ownerCheckVerifyConn(_ context.Context, local, remote net.Addr) error {
	return connOwnerCheck(local, remote)
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

// readProcTables reads /proc/net/tcp and /proc/net/tcp6. ok is false only
// when neither exists (no /proc socket table on this platform: the caller
// falls back to best effort). Any other read failure is an error, so a
// Linux host whose table cannot be read fails closed rather than open.
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

// connOwnerErr checks, for a connection that is already established but has
// not carried a byte, who owns the OTHER end: the server-side socket, whose
// local address is our remote and whose remote is our local, and which the
// kernel creates with the listener's uid. Unlike listenerOwnerErr this has no
// check-then-dial window: it is the very connection the token would go over.
// Same fail-closed rules: only a platform with no socket table returns nil
// unchecked.
func connOwnerErr(local, remote net.Addr, read func(string) ([]byte, error), myUID int) error {
	l, ok1 := local.(*net.TCPAddr)
	r, ok2 := remote.(*net.TCPAddr)
	if !ok1 || !ok2 {
		return fmt.Errorf("cannot verify the peer of a %s connection", remote.Network())
	}
	texts, ok, err := readProcTables(read)
	if err != nil {
		return err
	}
	if !ok {
		return nil // no socket table on this platform: best effort
	}
	for _, b := range texts {
		for _, s := range parseProcSockets(b) {
			if s.state == tcpListenState || s.localPort != r.Port || s.remPort != l.Port ||
				!s.localIP.Equal(r.IP) || !s.remIP.Equal(l.IP) {
				continue
			}
			if s.uid != myUID {
				return &portOwnerError{port: strconv.Itoa(r.Port), uid: s.uid}
			}
			return nil
		}
	}
	return fmt.Errorf("cannot verify the owner of %s: its end of the connection is not in the socket table", remote)
}
