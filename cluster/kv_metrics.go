// SPDX-License-Identifier: Apache-2.0

package cluster

import (
	"bytes"

	"github.com/rostamlabs/rostam/cache"
)

// handleKVMetrics renders THIS node's KV cache stats as Prometheus text,
// aggregated across every shard it hosts.
//
// It must be node-local. The registry's __kv_metrics__ handler reads the cache
// behind one TxContext, and in cluster mode a shardless op routes to shard 0 —
// so dispatching it through the normal path would report a single shard's cache
// and could forward the scrape to whichever node owns that shard, which is not
// the node the operator asked. Dispatching off n.adminOps (like __ready__ and
// __repl_metrics__) keeps the answer about the node that received it.
func (n *Node) handleKVMetrics(_ []byte) ([]byte, error) {
	var agg cache.Stats
	for _, s := range n.snapshotShards() {
		if s == nil {
			continue
		}
		agg.Add(s.Stats().Cache)
	}
	var buf bytes.Buffer
	if err := agg.WritePrometheus(&buf); err != nil {
		return nil, err
	}
	return buf.Bytes(), nil
}
