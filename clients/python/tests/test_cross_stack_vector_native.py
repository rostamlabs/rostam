"""Python<->Go cross-stack test for vector ops over the native TCP protocol,
through the unified ``Rostam`` facade's flat vector API.

The byte layouts are pinned by test_vecwire_golden; this proves the round trip:
a real server accepts each request and returns what we expect. It covers the
JSON-carrying parts (metadata, filter, content) that the golden test leaves to a
live server on purpose, and confirms create_collection actually builds a usable
collection for HNSW, Vamana and IVF.

Uses the public package API (``from rostam import Rostam, ...``) and calls the
flat API (``r.search(...)``, not ``r.vector.search(...)``). Return types are
the unified result objects (``SearchResult``/``Document``/``Group``/``Point``/
``ScrollPage``), accessed by attribute rather than by dict key.

Skipped when no server binary is found (same rule as the other cross-stack
modules): $ROSTAM_SERVER_BIN, or a `rostam-server*` built at the repo root.
"""

from __future__ import annotations

import socket
import subprocess
import tempfile
import time
import unittest

from _serverbin import find_server_bin
from rostam import Rostam, RostamError, TransportError, filters as f

_BIN, _WHY = find_server_bin()


def _free_port():
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


def _wait_tcp(host, port, deadline):
    while time.time() < deadline:
        try:
            socket.create_connection((host, port), timeout=0.5).close()
            return True
        except OSError:
            time.sleep(0.1)
    return False


@unittest.skipUnless(_BIN, _WHY)
class CrossStackVectorNativeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.http = _free_port()
        cls.tcp = _free_port()
        cls.dir = tempfile.mkdtemp(prefix="rostam-vnative-")
        cls.proc = subprocess.Popen(
            [_BIN, "-http", f"127.0.0.1:{cls.http}", "-tcp", f"127.0.0.1:{cls.tcp}", "-data", cls.dir],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        if not _wait_tcp("127.0.0.1", cls.tcp, time.time() + 20):
            cls.proc.kill()
            raise RuntimeError("rostam-server -tcp did not come up")
        cls.r = Rostam(f"tcp://127.0.0.1:{cls.tcp}")

    @classmethod
    def tearDownClass(cls):
        cls.r.close()
        cls.proc.terminate()
        try:
            cls.proc.wait(timeout=5)
        except Exception:
            cls.proc.kill()

    def test_create_upsert_search_get_delete(self):
        r = self.r
        r.create_collection("c", dim=4, metric="cosine")
        r.upsert("c", 1, [0.9, 0.1, 0, 0], content="alpha", metadata={"tenant": "acme"})
        r.upsert("c", 2, [0, 0, 0.9, 0.1], content="beta", metadata={"tenant": "beta"})

        hits = r.search("c", [0.9, 0.1, 0, 0], k=2)
        self.assertEqual(hits[0].id, 1)  # nearest to cluster A is id 1
        # Single-node server never emits a degraded trailer.
        self.assertIs(hits.degraded, False)
        self.assertEqual(hits.missing, [])

        got = r.get("c", 1)
        self.assertIsNotNone(got)
        self.assertEqual(got.metadata, {"tenant": "acme"})   # $content lifted out
        self.assertEqual(got.content, "alpha")
        self.assertEqual(len(got.vector), 4)

        self.assertTrue(r.exists("c", 1))
        self.assertFalse(r.exists("c", 99))
        self.assertIsNone(r.get("c", 99))                # miss -> None

        self.assertTrue(r.delete("c", 1))
        self.assertFalse(r.delete("c", 1))               # already gone
        self.assertIsNone(r.get("c", 1))

    def test_drop_collection_round_trips(self):
        # TcpTransport.drop_collection is new (Task 6): a real server must
        # accept the "vector_drop_collection" op with the encoded name and
        # actually remove the collection — a point write against it afterward
        # must fail, and a fresh create_collection under the same name must
        # start from empty (no leftover point).
        r = self.r
        r.create_collection("dc", dim=4, metric="cosine")
        r.upsert("dc", 1, [0.1, 0.2, 0.3, 0.4])
        self.assertTrue(r.exists("dc", 1))

        r.drop_collection("dc")
        with self.assertRaises(RostamError):
            r.upsert("dc", 2, [0.1, 0.2, 0.3, 0.4])   # collection no longer exists

        r.create_collection("dc", dim=4, metric="cosine")   # recreate under the same name
        self.assertFalse(r.exists("dc", 1))                 # empty, not the pre-drop data

    def test_metadata_filter_round_trips(self):
        r = self.r
        r.create_collection("f", dim=4, metric="cosine")
        r.upsert("f", 1, [0.9, 0.1, 0, 0], metadata={"tenant": "acme"})
        r.upsert("f", 2, [0.8, 0.2, 0, 0], metadata={"tenant": "beta"})
        hits = r.search("f", [0.9, 0.1, 0, 0], k=5, filter=f.eq("tenant", "beta"))
        self.assertEqual([h.id for h in hits], [2])

    def test_insert_is_create_only(self):
        r = self.r
        r.create_collection("i", dim=4, metric="cosine")
        r.insert("i", 1, [0.1, 0.2, 0.3, 0.4])
        self.assertTrue(r.exists("i", 1))
        with self.assertRaises(RostamError):
            r.insert("i", 1, [0.5, 0.5, 0.5, 0.5])   # id is live

    def test_sparse_round_trips(self):
        # The unified Point type (get()'s return) only exposes id/vector/
        # content/metadata across both transports — it does not carry sparse
        # components. This still exercises the sparse upsert wire path
        # end-to-end (encode -> live server -> stored) without asserting on
        # fields Point doesn't surface.
        r = self.r
        r.create_collection("sp", dim=4, metric="cosine")
        r.upsert("sp", 1, [0.1, 0.2, 0.3, 0.4],
                 sparse={"indices": [3, 17], "values": [0.8, 0.4]})
        got = r.get("sp", 1, with_payload=True)
        self.assertIsNotNone(got)
        self.assertEqual(len(got.vector), 4)

    def test_vamana_and_ivf_collections_build(self):
        r = self.r
        r.create_collection("vam", dim=4, metric="l2",
                             index_type="vamana", vamana_r=32, vamana_l=64)
        # IVF validates the graph params and defaults none of them (unlike
        # HNSW/Vamana), so m / ef_construction / ef_search are set explicitly —
        # the rule the docs call out for IVF tuning.
        r.create_collection("ivf", dim=4, metric="cosine",
                             m=16, ef_construction=200, ef_search=64,
                             index_type="ivf", ivf_nlist=16, ivf_nprobe=4)
        for col in ("vam", "ivf"):
            r.upsert(col, 1, [0.1, 0.2, 0.3, 0.4])
            self.assertTrue(r.exists(col, 1), col)

    def test_vector_ops_share_one_pooled_connection(self):
        r = self.r
        r.create_collection("mix", dim=4, metric="cosine")
        r.upsert("mix", 1, [0.1, 0.2, 0.3, 0.4], content="one")
        self.assertTrue(r.exists("mix", 1))
        got = r.get("mix", 1)
        self.assertEqual(got.content, "one")
        self.assertTrue(r.exists("mix", 1))

    def test_kv_and_vector_ops_share_one_pooled_connection(self):
        # r.kv and the flat vector API both call the same TcpTransport's
        # _call, so they share one connection pool and auth token — interleave
        # KV and vector ops on one client and confirm both work, proving the
        # pool is genuinely shared rather than each namespace opening its own.
        r = self.r
        r.kv.put("kv:key", b"val")
        self.assertEqual(r.kv.get("kv:key"), b"val")
        self.assertEqual(r.kv.incr("kv:ctr", 5), 5)
        # a vector op right after a KV op on the same pooled connection
        r.create_collection("mix2", dim=4, metric="cosine")
        r.upsert("mix2", 1, [0.1, 0.2, 0.3, 0.4])
        self.assertTrue(r.exists("mix2", 1))
        self.assertEqual(r.kv.get("kv:key"), b"val")
        self.assertTrue(r.kv.delete("kv:key"))

    # ---- Phase C: batch / scroll / RAG-shaped search / hybrid / recommend ----

    def test_get_batch(self):
        r = self.r
        r.create_collection("gb", dim=4, metric="cosine")
        r.upsert("gb", 1, [0.1, 0.2, 0.3, 0.4], content="one", metadata={"tenant": "a"})
        r.upsert("gb", 2, [0.5, 0.6, 0.7, 0.8], metadata={"tenant": "b"})
        points = r.get_batch("gb", [1, 2, 99])
        # get_batch returns one Point per PRESENT id (99 is absent, so
        # omitted) — the unified Point contract shared with the HTTP backend.
        self.assertEqual({p.id for p in points}, {1, 2})
        by_id = {p.id: p for p in points}
        self.assertEqual(len(by_id[1].vector), 4)
        self.assertEqual(by_id[1].content, "one")
        self.assertEqual(by_id[1].metadata, {"tenant": "a"})
        self.assertEqual(by_id[2].metadata, {"tenant": "b"})

    def test_get_batch_projection(self):
        r = self.r
        r.create_collection("gbp", dim=4, metric="cosine")
        r.upsert("gbp", 1, [0.1, 0.2, 0.3, 0.4], metadata={"a": 1})
        points = r.get_batch("gbp", [1], with_vector=False, with_payload=False)
        self.assertEqual(len(points), 1)
        self.assertIsNone(points[0].vector)
        self.assertEqual(points[0].metadata, {})

    def test_scroll_pages_through_all_points_with_cursor(self):
        r = self.r
        r.create_collection("sc", dim=4, metric="cosine")
        for i in range(1, 6):
            r.upsert("sc", i, [float(i)] * 4, content=f"doc{i}")
        seen = []
        cursor = ""
        for _ in range(10):                              # bounded loop guards against an infinite scroll
            page = r.scroll("sc", limit=2, cursor=cursor)
            seen.extend(d.id for d in page)
            cursor = page.next_cursor
            if not cursor:
                break
        self.assertEqual(sorted(seen), [1, 2, 3, 4, 5])
        self.assertEqual(cursor, "")                      # exhausted

    def test_scroll_filter(self):
        r = self.r
        r.create_collection("scf", dim=4, metric="cosine")
        r.upsert("scf", 1, [0.1, 0.2, 0.3, 0.4], metadata={"tenant": "acme"})
        r.upsert("scf", 2, [0.5, 0.6, 0.7, 0.8], metadata={"tenant": "beta"})
        page = r.scroll("scf", filter=f.eq("tenant", "beta"), limit=10)
        self.assertEqual([d.id for d in page], [2])
        self.assertEqual(page.next_cursor, "")

    def test_search_docs(self):
        r = self.r
        r.create_collection("sd", dim=4, metric="cosine")
        r.upsert("sd", 1, [0.9, 0.1, 0, 0], content="alpha", metadata={"tenant": "acme"})
        r.upsert("sd", 2, [0, 0, 0.9, 0.1], content="beta", metadata={"tenant": "beta"})
        docs = r.search_docs("sd", [0.9, 0.1, 0, 0], k=2)
        self.assertEqual(docs[0].id, 1)
        self.assertEqual(docs[0].content, "alpha")
        self.assertEqual(docs[0].metadata, {"tenant": "acme"})
        # Single-node server never emits a degraded trailer.
        self.assertIs(docs.degraded, False)
        self.assertEqual(docs.missing, [])

    def test_search_docs_filter(self):
        r = self.r
        r.create_collection("sdf", dim=4, metric="cosine")
        r.upsert("sdf", 1, [0.9, 0.1, 0, 0], content="alpha", metadata={"tenant": "acme"})
        r.upsert("sdf", 2, [0.8, 0.2, 0, 0], content="beta", metadata={"tenant": "beta"})
        docs = r.search_docs("sdf", [0.9, 0.1, 0, 0], k=5, filter=f.eq("tenant", "beta"))
        self.assertEqual([d.id for d in docs], [2])

    def test_search_groups(self):
        r = self.r
        r.create_collection("sg", dim=4, metric="cosine")
        r.upsert("sg", 1, [0.9, 0.1, 0, 0], content="a1", metadata={"cat": "x"})
        r.upsert("sg", 2, [0.8, 0.2, 0, 0], content="a2", metadata={"cat": "x"})
        r.upsert("sg", 3, [0, 0, 0.9, 0.1], content="b1", metadata={"cat": "y"})
        groups = r.search_groups("sg", [0.9, 0.1, 0, 0], k=5, group_by="cat", group_size=2)
        self.assertGreaterEqual(len(groups), 2)           # groups formed for both cat values
        by_key = {g.key: g for g in groups}
        self.assertIn("x", by_key)
        self.assertIn("y", by_key)
        self.assertEqual(len(by_key["x"].hits), 2)         # group_size=2 caps the "x" group at 2 hits
        self.assertEqual({h.id for h in by_key["x"].hits}, {1, 2})
        self.assertEqual(by_key["y"].hits[0].id, 3)

    def test_hybrid_search(self):
        r = self.r
        r.create_collection("hs", dim=4, metric="cosine")
        r.upsert("hs", 1, [0.9, 0.1, 0, 0], sparse={"indices": [1, 5], "values": [0.5, 0.3]})
        r.upsert("hs", 2, [0, 0, 0.9, 0.1], sparse={"indices": [2, 5], "values": [0.9, 0.1]})
        hits = r.hybrid_search("hs", [0.9, 0.1, 0, 0], k=2,
                                sparse={"indices": [1, 5], "values": [0.5, 0.3]})
        self.assertEqual(len(hits), 2)
        self.assertEqual(hits[0].id, 1)                    # dense+sparse both favor id 1
        self.assertGreater(hits[0].score, 0)
        # Single-node server never emits a degraded trailer.
        self.assertIs(hits.degraded, False)
        self.assertEqual(hits.missing, [])

    def test_hybrid_search_filter_and_weighted(self):
        r = self.r
        r.create_collection("hsf", dim=4, metric="cosine")
        r.upsert("hsf", 1, [0.9, 0.1, 0, 0], metadata={"tenant": "acme"})
        r.upsert("hsf", 2, [0.8, 0.2, 0, 0], metadata={"tenant": "beta"})
        hits = r.hybrid_search("hsf", [0.9, 0.1, 0, 0], k=5,
                                filter=f.eq("tenant", "beta"), method="weighted", alpha=0.5)
        self.assertEqual([h.id for h in hits], [2])

    def test_hybrid_text(self):
        r = self.r
        r.create_collection("ht", dim=4, metric="cosine", full_text=True)
        r.upsert("ht", 1, [0.9, 0.1, 0, 0], content="the quick brown fox jumps")
        r.upsert("ht", 2, [0, 0, 0.9, 0.1], content="a lazy dog sleeps all day")
        hits = r.hybrid_text("ht", [0.9, 0.1, 0, 0], "quick fox", k=2)
        self.assertEqual(len(hits), 2)
        self.assertEqual(hits[0].id, 1)                    # both dense and text lanes favor id 1
        self.assertGreater(hits[0].score, 0)
        # global_idf is new on TcpTransport.hybrid_text (Task 6, reconciled with
        # HttpTransport's identically-named kwarg) — a single-partition server
        # ignores it, but the wire flag must round-trip without erroring.
        gi_hits = r.hybrid_text("ht", [0.9, 0.1, 0, 0], "quick fox", k=2, global_idf=True)
        self.assertEqual([h.id for h in gi_hits], [h.id for h in hits])

    def test_recommend_excludes_seed_and_favors_similar(self):
        r = self.r
        r.create_collection("rc", dim=4, metric="cosine")
        r.upsert("rc", 1, [1, 0, 0, 0])
        r.upsert("rc", 2, [0.9, 0.1, 0, 0])   # close to the seed
        r.upsert("rc", 3, [0, 0, 1, 0])       # far from the seed
        recs = r.recommend("rc", [1], k=5)
        rec_ids = [x.id for x in recs]
        self.assertNotIn(1, rec_ids)                       # the positive seed itself is excluded
        self.assertEqual(rec_ids[0], 2)                    # nearest neighbour of the seed ranks first
        # Single-node server never emits a degraded trailer.
        self.assertIs(recs.degraded, False)
        self.assertEqual(recs.missing, [])

    def test_recommend_negative_and_filter(self):
        r = self.r
        r.create_collection("rcn", dim=4, metric="cosine")
        r.upsert("rcn", 1, [1, 0, 0, 0])
        r.upsert("rcn", 2, [0.9, 0.1, 0, 0], metadata={"tenant": "acme"})
        r.upsert("rcn", 3, [0, 0, 1, 0], metadata={"tenant": "beta"})
        recs = r.recommend("rcn", [1], k=5, filter=f.eq("tenant", "beta"))
        self.assertEqual([x.id for x in recs], [3])         # filter admits only the beta point

    def test_query_raises_on_tcp_use_recommend_instead(self):
        # The facade's r.query() is HTTP-only (TCP cannot build a general
        # QuerySpec) and must raise on a TCP client — see
        # tests/test_transport_gaps.py for the guard's unit coverage.
        # TcpTransport carries no query() of its own (dead code removed):
        # TCP clients use recommend() directly, which is exactly what the
        # old recommend-shaped query() forwarded to.
        r = self.r
        r.create_collection("qy", dim=4, metric="cosine")
        r.upsert("qy", 1, [1, 0, 0, 0])
        r.upsert("qy", 2, [0.9, 0.1, 0, 0])
        r.upsert("qy", 3, [0, 0, 1, 0])
        with self.assertRaises(TransportError):
            r.query("qy", [1], k=5)
        via_recommend = r.recommend("qy", [1], k=5)
        self.assertNotIn(1, [x.id for x in via_recommend])

    def test_upsert_batch(self):
        r = self.r
        r.create_collection("ub", dim=4, metric="cosine")
        r.upsert_batch("ub", [
            {"id": 1, "vector": [1, 0, 0, 0], "content": "p1"},
            {"id": 2, "vector": [0, 1, 0, 0], "metadata": {"k": "v"}},
            {"id": 3, "vector": [0, 0, 1, 0], "sparse": {"indices": [2], "values": [0.5]}},
        ])
        got1 = r.get("ub", 1)
        got2 = r.get("ub", 2)
        got3 = r.get("ub", 3)
        self.assertEqual(got1.content, "p1")
        self.assertEqual(got2.metadata, {"k": "v"})
        self.assertIsNotNone(got3)          # sparse upsert succeeded (Point doesn't expose sparse)
        points = r.get_batch("ub", [1, 2, 3])
        self.assertEqual({p.id for p in points}, {1, 2, 3})


if __name__ == "__main__":
    unittest.main()
