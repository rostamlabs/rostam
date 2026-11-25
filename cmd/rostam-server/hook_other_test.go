// SPDX-License-Identifier: Apache-2.0

//go:build !unix

package main

// Without flock no lock files are created, so pruning never meets one.
const flockSupported = false
