// SPDX-License-Identifier: Apache-2.0

//go:build !unix

package main

// checkListenerOwner is a no-op: there is no daemon on this platform, and no
// /proc to read the listener's owner from.
func checkListenerOwner(string) error { return nil }

// checkConnOwner is a no-op for the same reason.
func checkConnOwner(string) error { return nil }
