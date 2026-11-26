// SPDX-License-Identifier: Apache-2.0

package main

import (
	"os"
	"testing"
)

// TestMain points the namespace registry at a throwaway directory, so no test
// ever reads or writes the developer's ~/.rostam/memory/namespaces.json.
func TestMain(m *testing.M) {
	dir, err := os.MkdirTemp("", "rostam-nsregistry-")
	if err != nil {
		panic(err)
	}
	nsRegistryDir = func() (string, error) { return dir, nil }
	code := m.Run()
	_ = os.RemoveAll(dir)
	os.Exit(code)
}
