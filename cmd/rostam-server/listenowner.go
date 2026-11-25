// SPDX-License-Identifier: Apache-2.0

package main

import (
	"bufio"
	"bytes"
	"context"
	"encoding/binary"
	"encoding/hex"
	"fmt"
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

// listenerOwnerCheck verifies, before any token is sent to addr, that the
// process listening there runs as this user. It is a seam for tests; the real
// check is platform-specific (checkListenerOwner). Where the kernel's socket
// table can be read (Linux) it fails closed; where it cannot be read at all
// (other unixes) it returns nil and the caller proceeds as before.
var listenerOwnerCheck = checkListenerOwner

// ownerCheckBeforeConnect is listenerOwnerCheck as a client BeforeConnect
// hook, so the check runs before every dial, not only the first.
func ownerCheckBeforeConnect(_ context.Context, addr string) error {
	return listenerOwnerCheck(addr)
}

// procListener is one LISTEN socket from /proc/net/tcp or /proc/net/tcp6.
type procListener struct {
	ip  net.IP
	v6  bool // from tcp6
	uid int
}

const tcpListenState = "0A"

// parseProcNetTCP returns the LISTEN sockets on port from the text of
// /proc/net/tcp or /proc/net/tcp6. Lines that do not parse are skipped.
//
// Each line is "sl local rem st tx:rx tr:when retrnsmt uid ...". The local
// address is HEX_IP:HEX_PORT, the port big-endian, the IP printed as one
// (tcp) or four (tcp6) host-order 32-bit words.
func parseProcNetTCP(text []byte, port int) []procListener {
	var out []procListener
	sc := bufio.NewScanner(bytes.NewReader(text))
	for sc.Scan() {
		f := strings.Fields(sc.Text())
		if len(f) < 8 || f[3] != tcpListenState {
			continue
		}
		ipHex, portHex, ok := strings.Cut(f[1], ":")
		if !ok {
			continue
		}
		p, err := strconv.ParseUint(portHex, 16, 16)
		if err != nil || int(p) != port {
			continue
		}
		raw, err := hex.DecodeString(ipHex)
		if err != nil || (len(raw) != net.IPv4len && len(raw) != net.IPv6len) {
			continue
		}
		ip := make(net.IP, len(raw))
		for i := 0; i < len(raw); i += 4 {
			binary.NativeEndian.PutUint32(ip[i:i+4], binary.BigEndian.Uint32(raw[i:i+4]))
		}
		uid, err := strconv.Atoi(f[7])
		if err != nil || uid < 0 {
			continue
		}
		out = append(out, procListener{ip: ip, v6: len(raw) == net.IPv6len, uid: uid})
	}
	return out
}

// listenerOwnerErr is the decision behind checkListenerOwner: read is
// os.ReadFile, myUID os.Getuid(). It returns a *portOwnerError when the socket
// that would accept a connection to addr is owned by another uid, and nil when
// it is ours. It fails closed: an address that is not IP:port (a hostname
// could resolve anywhere, an empty host means every interface) and a readable
// socket table with no listener for addr are errors, since the owner of what
// we would connect to cannot be shown to be us. Only when neither table can be
// read at all (no /proc) does it return nil.
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
	var all []procListener
	readable := false
	for _, f := range []string{"/proc/net/tcp", "/proc/net/tcp6"} {
		if b, err := read(f); err == nil {
			readable = true
			all = append(all, parseProcNetTCP(b, port)...)
		}
	}
	if !readable {
		return nil // no socket table on this platform: best effort
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
