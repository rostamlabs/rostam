// SPDX-License-Identifier: Apache-2.0

package ops

import "errors"

// ####################### PLACEHOLDER — TASK 6 OWNS THIS ######################
//
// wire.BuiltinOps has carried a "kv_query" row since the codecs landed
// (sdk/wire/builtin_routing.go), and RegisterBuiltins refuses to register a
// builtin that has no handler — so from that commit until the query handler
// itself lands, EVERY caller of RegisterBuiltins failed outright: the whole ops
// suite, the whole shard suite, and every test that builds a store. This file
// is the smallest thing that closes that gap without pre-empting the design.
//
// It is deliberately not the real handler and deliberately not a silent
// success: it registers under the right name, with the right signature, and
// refuses. The record index it will read from is already maintained by every
// write handler (TxContext.reindexKV) and reachable through TxContext.KVIndex.
//
// Task 6 REPLACES this function body — the registry entry, the op name and the
// call shape all stay as they are, so nothing else has to move.
func handleKVQuery(_ *TxContext, _ []byte) ([]byte, error) {
	return nil, errKVQueryNotImplemented
}

var errKVQueryNotImplemented = errors.New("ops: kv_query: not implemented yet")
