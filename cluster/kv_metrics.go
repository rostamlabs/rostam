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
	// Hold shardMu across the whole read, rather than iterating a
	// snapshotShards() copy: that helper releases the lock before returning, so
	// a store it handed back can be closed by RemoveShardOwner while this loop
	// is still calling Stats() on it. (RemoveShardOwner nils the slot under the
	// write lock and closes AFTER releasing, so holding the read lock is what
	// keeps the store alive for the call.)
	//
	// CacheStatsNoWalk, not Stats: the full read recomputes reclaimable bytes
	// with an O(entries) walk on eligible mmap shards, and doing that under
	// shardMu would let a metrics scrape stall shard add/remove and node
	// shutdown for the length of the walk. A scrape takes the last published
	// figure instead.
	n.shardMu.RLock()
	for _, s := range n.shards {
		if s == nil {
			continue
		}
		agg.Add(s.CacheStatsNoWalk())
	}
	n.shardMu.RUnlock()
	var buf bytes.Buffer
	if err := agg.WritePrometheus(&buf); err != nil {
		return nil, err
	}
	return buf.Bytes(), nil
}
