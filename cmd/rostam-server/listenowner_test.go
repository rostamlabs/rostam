// SPDX-License-Identifier: Apache-2.0

package main

import (
	"context"
	"encoding/binary"
	"errors"
	"fmt"
	"net"
	"os"
	"strings"
	"testing"
	"time"

	"github.com/rostamlabs/rostam/ops"
)

const procHeader = "  sl  local_address rem_address   st tx_queue rx_queue tr tm->when retrnsmt   uid  timeout inode\n"

// procHex encodes ip the way /proc/net/tcp{,6} prints it: each 32-bit word
// of the address printed as a host-order integer. A v6 table entry is always
// 16 bytes, a v4-mapped one included (0000000000000000FFFF00000100007F).
func procHex(ip net.IP, v6 bool) string {
	b := ip.To4()
	if v6 || b == nil {
		b = ip.To16()
	}
	var s strings.Builder
	for i := 0; i < len(b); i += 4 {
		fmt.Fprintf(&s, "%08X", binary.NativeEndian.Uint32(b[i:i+4]))
	}
	return s.String()
}

func procLine(n int, ip string, port int, state string, uid int) string {
	h := procHex(net.ParseIP(ip), strings.Contains(ip, ":"))
	zero := strings.Repeat("0", len(h))
	return fmt.Sprintf("%4d: %s:%04X %s:0000 %s 00000000:00000000 00:00000000 00000000 %5d        0 4242 1 0000000000000000 100 0 0 10 0\n",
		n, h, port, zero, state, uid)
}

func ownerErr(tcp, tcp6 string, myUID int) error {
	files := map[string]string{"/proc/net/tcp": tcp, "/proc/net/tcp6": tcp6}
	read := func(p string) ([]byte, error) {
		if s, ok := files[p]; ok && s != "" {
			return []byte(s), nil
		}
		return nil, os.ErrNotExist
	}
	return listenerOwnerErr("127.0.0.1:7480", read, myUID)
}

// unverified is the wantUID for "fails closed: owner cannot be shown".
const unverified = -1

func TestListenerOwner(t *testing.T) {
	const me, other = 1000, 1001
	for _, c := range []struct {
		name      string
		tcp, tcp6 string
		wantUID   int // 0 = no error, unverified = a non-owner error
	}{
		{"ours", procHeader + procLine(0, "127.0.0.1", 7480, "0A", me), "", 0},
		{"other user", procHeader + procLine(0, "127.0.0.1", 7480, "0A", other), "", other},
		{"missing entry fails closed", procHeader + procLine(0, "127.0.0.1", 7481, "0A", other), "", unverified},
		{"empty tables fail closed", procHeader, procHeader, unverified},
		{"proc unreadable falls back", "", "", 0},
		{"not listening fails closed", procHeader + procLine(0, "127.0.0.1", 7480, "01", other), "", unverified},
		{"wildcard v4 other user", procHeader + procLine(0, "0.0.0.0", 7480, "0A", other), "", other},
		{"exact beats wildcard", procHeader + procLine(0, "0.0.0.0", 7480, "0A", other) + procLine(1, "127.0.0.1", 7480, "0A", me), "", 0},
		{"ipv6 mapped other user", procHeader, procHeader + procLine(0, "::ffff:127.0.0.1", 7480, "0A", other), other},
		{"ipv6 wildcard other user", procHeader, procHeader + procLine(0, "::", 7480, "0A", other), other},
		{"ipv6 loopback does not serve 127.0.0.1", procHeader, procHeader + procLine(0, "::1", 7480, "0A", other), unverified},
		{"ipv6 mapped ours", procHeader, procHeader + procLine(0, "::ffff:127.0.0.1", 7480, "0A", me), 0},
		{"malformed lines skipped", procHeader + "garbage\n   7: zz:yy 1 2\n 3: 0100007F:1D38 00000000:0000 0A x y z notanumber\n" + procLine(0, "127.0.0.1", 7480, "0A", me), "", 0},
	} {
		t.Run(c.name, func(t *testing.T) {
			err := ownerErr(c.tcp, c.tcp6, me)
			var pe *portOwnerError
			switch {
			case c.wantUID == unverified:
				if err == nil || errors.As(err, &pe) {
					t.Fatalf("err = %v, want a cannot-verify error", err)
				}
			case c.wantUID == 0 && err != nil:
				t.Fatalf("err = %v, want nil", err)
			case c.wantUID != 0 && (!errors.As(err, &pe) || pe.uid != c.wantUID):
				t.Fatalf("err = %v, want a portOwnerError for uid %d", err, c.wantUID)
			case c.wantUID != 0 && (!strings.Contains(err.Error(), "7480") || !strings.Contains(err.Error(), "one user per host")):
				t.Fatalf("message %q must name the port and the one-user-per-host limit", err)
			}
		})
	}
}

// The kernel's literal tcp6 form of a v4-mapped 127.0.0.1 listener.
func TestListenerOwnerLiteralMappedFixture(t *testing.T) {
	if binary.NativeEndian.Uint16([]byte{1, 0}) != 1 {
		t.Skip("fixture is little-endian")
	}
	tcp6 := procHeader +
		"   0: 0000000000000000FFFF00000100007F:1D38 00000000000000000000000000000000:0000 0A 00000000:00000000 00:00000000 00000000  1001        0 4242 1 0000000000000000 100 0 0 10 0\n"
	var pe *portOwnerError
	if err := ownerErr(procHeader, tcp6, 1000); !errors.As(err, &pe) || pe.uid != 1001 {
		t.Fatalf("err = %v; want uid 1001's mapped listener refused", err)
	}
	if err := ownerErr(procHeader, tcp6, 1001); err != nil {
		t.Fatalf("own mapped listener: %v", err)
	}
}

// Only an IP:port can be checked; anything else fails closed rather than
// sending the token unchecked.
func TestListenerOwnerNonIPHostFailsClosed(t *testing.T) {
	read := func(string) ([]byte, error) { return nil, os.ErrNotExist } // even with no /proc
	for _, addr := range []string{"localhost:7480", ":7480", "example.com:7480", "127.0.0.1", "127.0.0.1:http", "127.0.0.1:0"} {
		if err := listenerOwnerErr(addr, read, 1000); err == nil {
			t.Errorf("listenerOwnerErr(%q) = nil, want an error", addr)
		}
	}
}

// The client's BeforeConnect hook runs the owner check on every dial, so a
// long-lived -autostart session that redials never skips it.
func TestConnectStoreRunsOwnerCheckBeforeDial(t *testing.T) {
	orig := listenerOwnerCheck
	t.Cleanup(func() { listenerOwnerCheck = orig })
	var checked []string
	listenerOwnerCheck = func(addr string) error {
		checked = append(checked, addr)
		return &portOwnerError{port: "7480", uid: 7}
	}
	ln, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = ln.Close() })
	accepted := make(chan struct{}, 1)
	go func() {
		if c, err := ln.Accept(); err == nil {
			accepted <- struct{}{}
			_ = c.Close()
		}
	}()
	reg := ops.NewRegistry()
	if err := ops.RegisterBuiltins(reg); err != nil {
		t.Fatal(err)
	}
	st, err := connectStore(storeFlags{connect: ln.Addr().String(), authToken: "secret", beforeConnect: ownerCheckBeforeConnect}, reg,
		func(string) (string, bool) { return "", false })
	if err == nil {
		ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
		defer cancel()
		_, err = st.Get(ctx, []byte("k"))
		_ = st.Close()
	}
	var pe *portOwnerError
	if !errors.As(err, &pe) {
		t.Fatalf("err = %v, want the owner error from BeforeConnect", err)
	}
	if len(checked) == 0 || checked[0] != ln.Addr().String() {
		t.Fatalf("owner check saw %v, want %s", checked, ln.Addr())
	}
	select {
	case <-accepted:
		t.Fatal("a connection reached the listener despite the owner check")
	case <-time.After(100 * time.Millisecond):
	}
}

// The real kernel format on a little-endian host, as printed by Linux.
func TestListenerOwnerLiteralFixture(t *testing.T) {
	if binary.NativeEndian.Uint16([]byte{1, 0}) != 1 {
		t.Skip("fixture is little-endian")
	}
	tcp := procHeader +
		"   0: 0100007F:1D38 00000000:0000 0A 00000000:00000000 00:00000000 00000000  1001        0 233226867 1 0000000000000000 100 0 0 10 0\n"
	if err := ownerErr(tcp, "", 1000); err == nil {
		t.Fatal("uid 1001 owns 127.0.0.1:7480; want an error for uid 1000")
	}
	if err := ownerErr(tcp, "", 1001); err != nil {
		t.Fatalf("own listener: %v", err)
	}
}

func TestEnsureDaemonPortOwnedByOtherUser(t *testing.T) {
	data := t.TempDir()
	probe := func(ctx context.Context, addr, token string) error {
		return &portOwnerError{port: "7480", uid: 1234}
	}
	spawn := func() error { t.Error("spawn must not be called when another user owns the port"); return nil }
	err := ensureDaemonWith(context.Background(), data, "127.0.0.1:7480", spawn, probe, time.Second)
	if err == nil || !strings.Contains(err.Error(), "uid 1234") || !strings.Contains(err.Error(), "one user per host") {
		t.Fatalf("err = %v", err)
	}
}

func TestHookOpenChecksOwnerBeforeToken(t *testing.T) {
	orig := listenerOwnerCheck
	t.Cleanup(func() { listenerOwnerCheck = orig })
	listenerOwnerCheck = func(string) error { return &portOwnerError{port: "7480", uid: 7} }
	tokenFile := t.TempDir() + "/token"
	if err := os.WriteFile(tokenFile, []byte("secret\n"), 0o600); err != nil {
		t.Fatal(err)
	}
	st, err := hookOpen("127.0.0.1:1", tokenFile)(context.Background())
	var pe *portOwnerError
	if st != nil || !errors.As(err, &pe) {
		t.Fatalf("hookOpen = %v, %v; want the owner error before any connection", st, err)
	}
}

func TestProbeDaemonChecksOwner(t *testing.T) {
	ln, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = ln.Close() })
	orig := listenerOwnerCheck
	t.Cleanup(func() { listenerOwnerCheck = orig })
	var checked string
	listenerOwnerCheck = func(addr string) error { checked = addr; return &portOwnerError{port: "x", uid: 9} }
	var pe *portOwnerError
	if err := probeDaemon(context.Background(), ln.Addr().String(), "tok"); !errors.As(err, &pe) || checked != ln.Addr().String() {
		t.Fatalf("probeDaemon = %v (checked %q); want the owner error for %s", err, checked, ln.Addr())
	}
}
