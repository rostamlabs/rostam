"""Python<->Go cross-stack smoke for the Query API (query/recommend/discover).

recommend and discover have no standalone HTTP route and no engine-fake stand-in
worth trusting: they are leaves of the unified Query API, and their whole value
is the RANKING the real engine produces. A fake that returns a canned order
would prove the client talks to itself, not that these methods retrieve the
right neighbours. So this launches the real rostam-server and asserts on the
structure of the results — same-cluster ids come back, the negative example is
pushed away, the metadata filter is honoured — rather than on a fixed order the
engine is free to change.

The corpus is two well-separated clusters so "did it retrieve the right region"
is unambiguous. The server binary is found via $ROSTAM_SERVER_BIN or a
`rostam-server*` built at the repo root; the module is skipped when none exists.
"""

from __future__ import annotations

import socket
import subprocess
import tempfile
import time
import unittest

from _serverbin import find_server_bin
from rostam import Rostam, filters as f

DIM = 4
# Cluster A ids 1-3 near [1,0,0,0]; cluster B ids 4-6 near [0,0,1,0].
CORPUS = {
    1: [0.98, 0.02, 0.0, 0.0], 2: [0.95, 0.05, 0.0, 0.0], 3: [0.92, 0.08, 0.0, 0.0],
    4: [0.0, 0.0, 0.98, 0.02], 5: [0.0, 0.0, 0.95, 0.05], 6: [0.0, 0.0, 0.92, 0.08],
}
CLUSTER_A = {1, 2, 3}
CLUSTER_B = {4, 5, 6}


def _free_port():
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


_BIN, _WHY = find_server_bin()


@unittest.skipUnless(_BIN, _WHY)
class CrossStackQueryTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.port = _free_port()
        cls.datadir = tempfile.mkdtemp(prefix="rostam-query-")
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

        cls.c.create_collection("q", dim=DIM, metric="cosine")
        for i, v in CORPUS.items():
            cls.c.upsert("q", i, v, content=f"pt{i}",
                         metadata={"grp": ("a" if i in CLUSTER_A else "b")})

    @classmethod
    def tearDownClass(cls):
        cls.c.close()
        cls.proc.terminate()
        try:
            cls.proc.wait(timeout=5)
        except Exception:
            cls.proc.kill()

    def test_query_single_dense_lane(self):
        hits = self.c.query("q", [{"dense": CORPUS[1]}], k=3)
        self.assertTrue(hits, "query returned no hits")
        # Nearest to cluster A's centre must be cluster A.
        self.assertIn(hits[0].id, CLUSTER_A)

    def test_query_requires_a_prefetch_leaf(self):
        with self.assertRaises(ValueError):
            self.c.query("q", [], k=3)

    def test_recommend_pulls_toward_positive_cluster(self):
        hits = self.c.recommend("q", positive=[1, 2], k=3)
        ids = [h.id for h in hits]
        self.assertTrue(ids, "recommend returned no hits")
        # The example ids are excluded from their own results, so of cluster A
        # only id 3 remains — and it must rank first, ahead of any cluster-B
        # point. (Asserting "majority A" would be wrong: A has three members and
        # two are the query, so the ranking cannot contain more than one A hit.)
        self.assertEqual(hits[0].id, 3)
        self.assertNotIn(1, ids)
        self.assertNotIn(2, ids)

    def test_recommend_negative_pushes_away(self):
        # Toward 1 (cluster A), away from 4 (cluster B): the top hit stays in A.
        hits = self.c.recommend("q", positive=[1], negative=[4], k=3)
        self.assertTrue(hits)
        self.assertIn(hits[0].id, CLUSTER_A)

    def test_recommend_honours_filter(self):
        # Score toward cluster A but restrict to grp=b — every hit must be in B.
        hits = self.c.recommend("q", positive=[1], k=3, filter=f.eq("grp", "b"))
        self.assertTrue(hits, "filtered recommend returned no hits")
        self.assertTrue(all(h.id in CLUSTER_B for h in hits))

    def test_discover_explores_positive_region(self):
        # Context (positive=5 in B, negative=1 in A) → cluster B.
        hits = self.c.discover("q", context=[(5, 1)], k=3, target=5)
        self.assertTrue(hits, "discover returned no hits")
        self.assertIn(hits[0].id, CLUSTER_B)


if __name__ == "__main__":
    unittest.main()
