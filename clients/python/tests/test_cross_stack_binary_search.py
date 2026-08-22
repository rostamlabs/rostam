"""Python<->Go cross-stack smoke for the binary query wire (RVQ1).

Same reasoning as the bulk-wire test next door: the framing is a byte-level
contract between two languages, and a fake proves only that the client agrees
with itself. This launches the REAL rostam-server and asserts EQUIVALENCE — the
same query sent in both encodings must come back with identical hits, in the
same order, with identical distances. Anything less would let the binary path
quietly become a second search semantics.

A byte-order or f32/f64 slip does not usually produce an error; it produces a
plausible ranking of the wrong neighbours. So the vectors below are chosen so
that every point's nearest neighbour is itself, and the assertions compare
distances rather than counts.

The server binary is located via $ROSTAM_SERVER_BIN, or a `rostam-server*` built
at the repo root. The whole module is skipped when no binary is found.
"""

from __future__ import annotations

import socket
import subprocess
import tempfile
import time
import unittest

from _serverbin import find_server_bin
from rostam import Rostam, RostamError

DIM = 24
N = 150


def _free_port():
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def _vec(i):
    return [i * 0.041 - 2.7 + d * 0.013 for d in range(DIM)]


_BIN, _WHY = find_server_bin()


@unittest.skipUnless(_BIN, _WHY)
class CrossStackBinarySearchTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.port = _free_port()
        cls.datadir = tempfile.mkdtemp(prefix="rostam-binsearch-")
        cls.proc = subprocess.Popen(
            [_BIN, "-http", f"127.0.0.1:{cls.port}", "-data", cls.datadir],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        cls.base = f"http://127.0.0.1:{cls.port}"
        cls.c = Rostam(cls.base, timeout=120)
        deadline = time.time() + 20
        while time.time() < deadline:
            if cls.proc.poll() is not None:
                raise RuntimeError("rostam-server exited during startup")
            try:
                if cls.c.health():
                    break
            except Exception:
                time.sleep(0.1)
        else:
            cls.proc.kill()
            raise RuntimeError("rostam-server did not become healthy in time")

        cls.c.create_collection("q", dim=DIM, metric="l2")
        for i in range(1, N + 1):
            cls.c.upsert("q", i, _vec(i), content=f"doc {i}", metadata={"bucket": i % 5})

    @classmethod
    def tearDownClass(cls):
        cls.c.close()
        cls.proc.terminate()
        try:
            cls.proc.wait(timeout=5)
        except Exception:
            cls.proc.kill()

    def _json_client(self):
        # binary_search isn't part of the unified Rostam() constructor surface
        # (the facade doesn't expose transport-tuning kwargs beyond pool_maxsize);
        # the underlying HttpTransport still has the attribute, so it is set
        # directly to force the JSON fallback path for this comparison.
        c = Rostam(self.base, timeout=120)
        c._t.binary_search = False
        return c

    def test_binary_query_matches_json_query(self):
        q = _vec(42)
        jc = self._json_client()
        try:
            bin_hits = self.c.search("q", q, k=10)
            json_hits = jc.search("q", q, k=10)
        finally:
            jc.close()

        self.assertTrue(bin_hits, "binary query returned no hits")
        self.assertEqual([h.id for h in json_hits], [h.id for h in bin_hits])
        for b, j in zip(bin_hits, json_hits):
            self.assertEqual(j.distance, b.distance,
                             f"distance differs for id {b.id}: binary {b.distance}, JSON {j.distance}")
        # The nearest neighbour of a stored point is that point itself; if the
        # framing mangled the vector this is the first thing to break.
        self.assertEqual(42, bin_hits[0].id)

    def test_binary_query_with_filter_matches_json(self):
        q = _vec(70)
        filt = {"op": "eq", "field": "bucket", "value": {"kind": "int", "int": 3}}
        jc = self._json_client()
        try:
            bin_hits = self.c.search("q", q, k=5, filter=filt)
            json_hits = jc.search("q", q, k=5, filter=filt)
        finally:
            jc.close()

        self.assertTrue(bin_hits, "filtered binary query returned no hits")
        self.assertEqual([h.id for h in json_hits], [h.id for h in bin_hits])
        for h in bin_hits:
            self.assertEqual(3, h.id % 5, f"filter not applied: id {h.id}")

    def test_search_docs_matches_json_over_the_binary_wire(self):
        q = _vec(99)
        jc = self._json_client()
        try:
            bin_docs = self.c.search_docs("q", q, k=5)
            json_docs = jc.search_docs("q", q, k=5)
        finally:
            jc.close()

        self.assertEqual([d.id for d in json_docs], [d.id for d in bin_docs])
        self.assertEqual([d.content for d in json_docs], [d.content for d in bin_docs])
        self.assertEqual("doc 99", bin_docs[0].content)

    def test_server_rejects_a_bad_k_over_the_binary_wire(self):
        """A request error must arrive as one, not as a silent fallback to JSON."""
        with self.assertRaises(RostamError) as caught:
            self.c.search("q", _vec(1), k=0)
        self.assertEqual(400, caught.exception.status)
        # Still on the binary path — a real 400 must not disable the framing.
        self.assertTrue(self.c._t._binary_search_supported)


if __name__ == "__main__":
    unittest.main()
