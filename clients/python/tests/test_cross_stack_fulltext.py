"""Python<->Go cross-stack smoke for the BM25 full-text surface.

It launches the REAL rostam-server binary (REST) and drives it end to end via
the Rostam facade: create a full_text collection -> upsert content docs -> search_text
(rare term ranks its doc first) -> hybrid_text (dense + BM25 fused). This proves
the Python SDK's raw-text requests round-trip through the live Go server, which
tokenizes + BM25-scores them server-side (the SDK ships no tokens).

The server binary is located via $ROSTAM_SERVER_BIN, or a `rostam-server*` built
at the repo root (e.g. `rostam-server-test`). The whole module is skipped when no
binary is found, so the unit suite still runs without a Go toolchain.
"""

from __future__ import annotations

import socket
import subprocess
import tempfile
import time
import unittest

from _serverbin import find_server_bin
from rostam import Rostam




def _free_port():
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


_BIN, _WHY = find_server_bin()


@unittest.skipUnless(_BIN, _WHY)
class CrossStackFullTextTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.port = _free_port()
        cls.datadir = tempfile.mkdtemp(prefix="rostam-ft-")
        cls.proc = subprocess.Popen(
            [_BIN, "-http", f"127.0.0.1:{cls.port}", "-data", cls.datadir],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        cls.base = f"http://127.0.0.1:{cls.port}"
        cls.c = Rostam(cls.base)
        # Wait for readiness (health) with a deadline.
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

    @classmethod
    def tearDownClass(cls):
        cls.proc.terminate()
        try:
            cls.proc.wait(timeout=5)
        except Exception:
            cls.proc.kill()

    def test_full_text_round_trip(self):
        docs = {
            1: "the quick brown fox jumps over the lazy dog",
            2: "the lazy dog sleeps all day in the sun",
            3: "a dog and another dog play in the park",
            4: "machine learning models rank documents by relevance",
            5: "brown bears and brown foxes roam the brown forest",
        }

        def dense_for(i):
            return [i * 0.01 + 0.1, (i % 5) * 0.2 + 0.05, (i % 7) * 0.13 + 0.02, (i % 3) * 0.31 + 0.07]

        self.c.create_collection("ft", dim=4, metric="l2", full_text=True)
        for i, content in docs.items():
            self.c.upsert("ft", i, dense_for(i), content=content)

        # search_text: the rare term "fox" ranks doc 1 first (doc 5 "foxes" stems too).
        res = self.c.search_text("ft", "fox", k=5)
        self.assertTrue(res, "search_text returned no documents")
        self.assertEqual(res[0].id, 1)
        self.assertEqual(res[0].content, docs[1])
        for d in res:
            self.assertIn(d.id, (1, 5))  # only fox-bearing docs

        # global_idf=True opts into the two-phase global-DF (dfs) path. On a
        # single-partition collection the flag is honored harmlessly (local corpus IS
        # global), so it round-trips through the live server and returns the SAME
        # ranking as the local-IDF path — proving the SDK->REST->coordinator wiring.
        gres = self.c.search_text("ft", "fox", k=5, global_idf=True)
        self.assertEqual([d.id for d in gres], [d.id for d in res])

        # hybrid_text: dense(doc1) + "fox" -> doc 1 first.
        hres = self.c.hybrid_text("ft", dense_for(1), "fox", k=5, method="rrf")
        self.assertTrue(hres, "hybrid_text returned no results")
        self.assertEqual(hres[0].id, 1)

        # hybrid_text with global_idf=True round-trips identically on P==1.
        ghres = self.c.hybrid_text("ft", dense_for(1), "fox", k=5, method="rrf", global_idf=True)
        self.assertEqual([r.id for r in ghres], [r.id for r in hres])

        # A text search on a NON-full-text collection is a 400 (RostamError).
        self.c.create_collection("plain", dim=4, metric="l2")
        self.c.upsert("plain", 1, dense_for(1), content="hello world")
        with self.assertRaises(Exception):
            self.c.search_text("plain", "hello", k=5)


if __name__ == "__main__":
    unittest.main()
