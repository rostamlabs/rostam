// SPDX-License-Identifier: Apache-2.0

package server

// Dispatcher is the minimal interface a server uses to dispatch frames
// into a backing store. Both *shard.Store and *cluster.Node satisfy it.
//
// Call runs an op by name and returns the encoded result bytes (or an
// error that the server maps to a wire status code).
//
// LeaderAddr returns an address of a current Raft leader, or an empty
// string when no leader is known. For *shard.Store this is the leader
// of its single Raft group. For *cluster.Node (multi-node)
// this is shard 0's leader — used as a fallback when the per-shard
// hint from *shard.NotLeaderError is absent.
//
// Per-shard NotLeader hints are the primary mechanism: server's
// mapResult prefers *shard.NotLeaderError.LeaderAddr from the
// dispatched Call's error chain. LeaderAddr() is only consulted
// when the wrapped hint is empty.
type Dispatcher interface {
	Call(name string, args []byte) ([]byte, error)
	LeaderAddr() string
}

// AppendDispatcher is the OPTIONAL allocation-free twin of Dispatcher. A
// dispatcher that implements it can serve a call into a buffer the transport
// owns, so a reply costs no allocation at all.
//
// The returned payload MAY alias dst — an op with an append variant writes into
// it, one without returns its own freshly built reply, and copying the latter
// into dst just to make the aliasing uniform would add a full reply copy to
// nearly every op. When it does alias, it is valid only until dst is reused,
// which is why this is opt-in rather than part of Dispatcher: only a transport
// that copies the payload out before its next call on that connection may ask
// for it. A caller reusing dst must therefore not ASSUME the result aliases it:
// dispatchInto reports whether the append path ran, which is the signal to keep
// the returned buffer — a payload that grew past dst is a new array and is the
// one most worth keeping.
// The epoll server qualifies — epollConn.encode copies the payload into the
// response frame before the loop reads the next request. A dispatcher that does
// not implement this is served through Call exactly as before.
type AppendDispatcher interface {
	// CallAppend serves the op, using dst when the op has an append handler.
	// appended reports whether that handler actually ran: an op WITHOUT one is
	// served normally and returns its own reply, which never came out of dst and
	// must not be retained as the connection's buffer. Reporting it here rather
	// than inferring it from success is what keeps a transport from adopting a
	// buffer it cannot reuse.
	CallAppend(name string, args, dst []byte) (payload []byte, appended bool, err error)
}
