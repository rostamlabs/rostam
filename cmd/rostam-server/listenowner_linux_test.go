// SPDX-License-Identifier: Apache-2.0

//go:build linux

package main

import (
	"errors"
	"net"
	"os"
	"testing"
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

// The post-dial check finds the server-side socket of the very connection,
// both while it waits in the accept backlog and once accepted, and refuses it
// for any other uid.
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
		if err := checkConnOwner(c.LocalAddr(), c.RemoteAddr()); err != nil {
			t.Fatalf("%s: own connection refused: %v", stage, err)
		}
		var pe *portOwnerError
		if err := connOwnerErr(c.LocalAddr(), c.RemoteAddr(), os.ReadFile, os.Getuid()+1); !errors.As(err, &pe) || pe.uid != os.Getuid() {
			t.Fatalf("%s: server end not found in /proc/net/tcp (err %v)", stage, err)
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
