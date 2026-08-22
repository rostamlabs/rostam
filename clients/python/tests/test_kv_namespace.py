"""Unit tests for the r.kv namespace wiring on the unified Rostam facade.

No network needed: construction is I/O-free on both backends, so this only
proves that the HTTP backend's r.kv is a sentinel that raises on first
attribute access. The live TCP round-trip (r.kv.get/put/...) is covered by
tests/test_cross_stack_kv.py against a real server.
"""

from __future__ import annotations

import unittest

from rostam._kv import _KV, _KVUnavailable
from rostam._tcp import TcpTransport
from rostam._types import TransportError
from rostam.rostam import Rostam


class KVNamespaceTest(unittest.TestCase):
    def test_kv_unavailable_on_http_raises(self):
        r = Rostam("http://127.0.0.1:8080")   # no connection needed to build
        self.assertIsInstance(r.kv, _KVUnavailable)
        with self.assertRaises(TransportError):
            r.kv.get("k")

    def test_kv_unavailable_raises_on_any_attribute_not_just_get(self):
        r = Rostam("http://127.0.0.1:8080")
        with self.assertRaises(TransportError):
            r.kv.put("k", "v")
        with self.assertRaises(TransportError):
            r.kv.ping()

    def test_kv_is_wired_and_shares_the_tcp_transport_on_tcp(self):
        # No connection needed to build: the socket pool connects lazily.
        r = Rostam("tcp://127.0.0.1:7000")
        self.assertIsInstance(r.kv, _KV)
        # r.kv reuses r._t (the TcpTransport) rather than opening a second pool.
        self.assertIs(r.kv._t, r._t)
        self.assertIsInstance(r._t, TcpTransport)


if __name__ == "__main__":
    unittest.main()
