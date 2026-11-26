// SPDX-License-Identifier: Apache-2.0

package main

import (
	"slices"
	"testing"
)

func TestSplitCollections(t *testing.T) {
	for _, tc := range []struct {
		in   string
		want []string
	}{
		{"", nil},
		{" , ,", nil},
		{"mcp_memory", []string{"mcp_memory"}},
		{" a, ,b ", []string{"a", "b"}},
		{"acme/notes,mcp_memory", []string{"acme/notes", "mcp_memory"}},
	} {
		got := splitCollections(tc.in)
		if !slices.Equal(got, tc.want) || (got == nil) != (tc.want == nil) {
			t.Errorf("splitCollections(%q) = %#v, want %#v", tc.in, got, tc.want)
		}
	}
}
