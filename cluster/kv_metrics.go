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
// The read lock is held across the WHOLE loop, and the shards must be read from
// n.shards under it rather than from a snapshotShards() copy. Both shutdown
// paths close a store the caller may still be holding: RemoveShardOwner nils the
// slot under the write lock and closes after releasing, and Node.Close closes
// every store IN PLACE without nilling anything — so after it runs the slots are
// non-nil and unmapped. Releasing the lock between shards would let either land
// mid-iteration and turn the next CacheStats into a read of unmapped memory.
//
// The cost is that a scrape can delay shard add/remove and node shutdown, since
// Stats recomputes reclaimable bytes with an O(entries) walk. That exposure is
// narrow and bounded: only an mmap + replicated + reject-writes shard walks at
// all (onlineCompactionEligible — every other configuration returns 0 without
// touching an entry), and reclaimableStatsTTL rate-limits it to once per 2s per
// shard. Serving the gauge from a cache instead is not an option: nothing
// publishes that cache unless OnlineCompaction is on, which it is not by
// default, so the scrape would report a permanent 0.
func (n *Node) handleKVMetrics(_ []byte) ([]byte, error) {
	var agg cache.Stats
	n.shardMu.RLock()
	for _, s := range n.shards {
		if s == nil {
			continue
		}
		agg.Add(s.CacheStats())
	}
	n.shardMu.RUnlock()
	var buf bytes.Buffer
	if err := agg.WritePrometheus(&buf); err != nil {
		return nil, err
	}
	return buf.Bytes(), nil
}
