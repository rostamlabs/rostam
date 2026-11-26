// SPDX-License-Identifier: Apache-2.0

//go:build linux

package main

import (
	"errors"
	"net"
	"os"
	"testing"
	"time"
)

// Against the real kernel table: our own listener is found and passes, and the
// same entry seen by a different uid is refused.
func TestListenerOwnerLiveProc(t *testing.T) {
	ln, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = ln.Close() })
	addr := ln.Addr().String()
	if err := checkListenerOwner(addr); err != nil {
		t.Fatalf("own listener refused: %v", err)
	}
	var pe *portOwnerError
	if err := listenerOwnerErr(addr, os.ReadFile, os.Getuid()+1); !errors.As(err, &pe) || pe.uid != os.Getuid() {
		t.Fatalf("listener not found in /proc/net/tcp (err %v)", err)
	}
}

// The post-dial check finds the server-side socket of our connection, both
// while it waits in the accept backlog and once accepted.
func TestConnOwnerLiveProc(t *testing.T) {
	ln, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = ln.Close() })
	c, err := net.Dial("tcp", ln.Addr().String())
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = c.Close() })
	check := func(stage string) {
		if err := checkConnOwner(ln.Addr().String()); err != nil {
			t.Fatalf("%s: own connection refused: %v", stage, err)
		}
		// Seen as another uid, our connection is not "ours", so nothing can
		// be verified: an error, never a pass.
		if err := connOwnerErr(ln.Addr().String(), os.ReadFile, os.Getuid()+1); err == nil {
			t.Fatalf("%s: verified with the wrong uid", stage)
		}
	}
	check("before accept")
	sc, err := ln.Accept()
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = sc.Close() })
	check("after accept")
}

// Against the real kernel: a connection the server half-closed (CLOSE_WAIT on
// our side) is still checked and passes when the server is ours, and once the
// server closes its end fully the stale connection does not cause a refusal.
func TestConnOwnerLiveHalfClosed(t *testing.T) {
	ln, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = ln.Close() })
	c, err := net.Dial("tcp", ln.Addr().String())
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = c.Close() })
	sc, err := ln.Accept()
	if err != nil {
		t.Fatal(err)
	}
	if err := sc.(*net.TCPConn).CloseWrite(); err != nil {
		t.Fatal(err)
	}
	buf := make([]byte, 1)
	if _, err := c.Read(buf); err == nil {
		t.Fatal("expected EOF after the server's half-close")
	}
	if err := checkConnOwner(ln.Addr().String()); err != nil {
		t.Fatalf("half-closed own connection refused: %v", err)
	}
	_ = sc.Close()
	time.Sleep(50 * time.Millisecond)
	if err := checkConnOwner(ln.Addr().String()); err != nil {
		t.Fatalf("connection whose server end is closed refused: %v", err)
	}
}
