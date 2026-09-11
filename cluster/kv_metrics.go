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
//
// The lock is taken PER SHARD rather than once around the loop. Stats on an
// eligible mmap shard can recompute reclaimable bytes with an O(entries) walk,
// and RemoveShardOwner nils the slot under the write lock before closing the
// store — so the read lock is what keeps a store alive across the call, but
// holding it across every shard would let one scrape stall shard add/remove and
// node shutdown for the sum of all their walks. Per shard, a waiting writer gets
// in at the next release, after at most one walk. The cost is that the shards
// are not sampled at one instant, which a counter scrape can afford.
func (n *Node) handleKVMetrics(_ []byte) ([]byte, error) {
	var agg cache.Stats
	for i := 0; ; i++ {
		n.shardMu.RLock()
		if i >= len(n.shards) {
			n.shardMu.RUnlock()
			break
		}
		var st cache.Stats
		if s := n.shards[i]; s != nil {
			st = s.CacheStats()
		}
		n.shardMu.RUnlock()
		agg.Add(st)
	}
	var buf bytes.Buffer
	if err := agg.WritePrometheus(&buf); err != nil {
		return nil, err
	}
	return buf.Bytes(), nil
}
