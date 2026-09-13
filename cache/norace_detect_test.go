// SPDX-License-Identifier: Apache-2.0

//go:build !race

package cache

// raceEnabled is false in normal (non -race) builds.
const raceEnabled = false
