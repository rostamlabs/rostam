// SPDX-License-Identifier: Apache-2.0

//go:build race

package ops

// raceEnabled is true in -race builds. The race detector's own sync.Pool.Put
// randomly drops ~1/4 of items on the floor (see sync/pool.go) specifically
// so pool-reliant code cannot assume retention across a Get/Put pair — which
// is exactly what the operate allocation-budget tests rely on. See
// TestDynamicEngineAllocs and TestSchemaEngineAllocs.
const raceEnabled = true
