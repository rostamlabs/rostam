"""Client tests against a stdlib HTTP fake that mimics the Rostam REST surface.

These run a real loopback HTTP server, so they exercise the client's request
construction (paths, JSON bodies, bearer auth, tagged-metadata encoding) and
response parsing (tagged-metadata/group decoding) over an actual socket — no
third-party deps required.

Uses ``from rostam.rostam import Rostam`` (not ``from rostam import Rostam``,
which still resolves to the pre-unification ``kv.Rostam`` until the old
classes are removed in a later task) and calls the flat vector API through the
facade, which delegates to ``rostam._http.HttpTransport``. Errors and result
types come from ``rostam._types`` (the unified set), not ``rostam.client``'s
now-superseded dataclasses.
"""

import json
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlsplit

from rostam import filters as f
from rostam._types import Point, RostamError, ScrollPage
from rostam.rostam import Rostam
from _wire import read_body
from _fakestore import FakeRostam

# Captured requests, newest last; reset per test.
REQUESTS = []


class _Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):  # silence test output
        pass

    def _body(self):
        return read_body(self.headers, self.rfile)

    def _record(self):
        body = self._body()
        REQUESTS.append({
            "method": self.command,
            "path": self.path,
            "auth": self.headers.get("Authorization"),
            "body": body,
        })
        return body

    def _send(self, code, obj):
        payload = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self):
        self._record()
        parsed = urlsplit(self.path)
        if parsed.path == "/v1/health":
            return self._send(200, {"status": "ok"})
        if "/points/" in parsed.path:
            # Match on the parsed path, not the raw self.path: a real GET
            # (Rostam.get()) always appends `?with_vector=...&with_payload=...`,
            # so `self.path.endswith("/missing")` would never fire for an
            # actual missing-point request — only for a query-string-free path.
            if parsed.path.endswith("/missing"):
                return self._send(404, {"error": "not found"})
            qs = parse_qs(parsed.query)
            with_vector = qs.get("with_vector", ["true"])[0] == "true"
            body = {"id": 1, "payload": {}}
            if with_vector:
                body["vector"] = [1.0, 0.0, 0.0]
            return self._send(200, body)
        self._send(404, {"error": "not found"})

    def do_DELETE(self):
        self._record()
        if self.path.endswith("/missing"):
            return self._send(200, {"deleted": False})
        if "/points/" in self.path:
            return self._send(200, {"deleted": True})
        self._send(200, {"dropped": "docs"})

    def do_POST(self):
        body = self._record()
        p = self.path
        if p == "/v1/collections":
            if body["config"].get("metric") == "bogus":
                return self._send(400, {"error": "unknown metric"})
            return self._send(201, {"name": body["name"]})
        if p.endswith("/points"):
            return self._send(200, {"id": body["id"]})
        if p.endswith("/points/delete"):
            return self._send(200, {"deleted": 2})
        if p.endswith("/search"):
            return self._send(200, {"results": [{"id": 1, "distance": 0.5, "score": 0.0}]})
        if p.endswith("/search/docs"):
            return self._send(200, {"documents": [
                {"id": 1, "distance": 0.5, "content": "hello",
                 "metadata": {"doc_id": {"kind": "int", "int": 7}, "tag": {"kind": "string", "str": "x"}}},
            ]})
        if p.endswith("/search/groups"):
            return self._send(200, {"groups": [
                {"key": {"kind": "int", "int": 1},
                 "hits": [{"id": 1, "distance": 0.5, "content": "c", "metadata": {}}]},
            ]})
        if p.endswith("/search/hybrid"):
            return self._send(200, {"results": [{"id": 2, "distance": 0.1, "score": 0.9}]})
        if p.endswith("/search/text"):
            return self._send(200, {"documents": [
                {"id": 4, "distance": 0.0, "score": 3.2, "content": "the quick brown fox",
                 "metadata": {"tag": {"kind": "string", "str": "y"}}},
            ]})
        if p.endswith("/search/hybrid-text"):
            return self._send(200, {"results": [{"id": 4, "distance": 0.2, "score": 0.8}]})
        self._send(404, {"error": "not found"})


class ClientTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.srv = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
        cls.thread = threading.Thread(target=cls.srv.serve_forever, daemon=True)
        cls.thread.start()
        host, port = cls.srv.server_address
        cls.base = f"http://{host}:{port}"

    @classmethod
    def tearDownClass(cls):
        cls.srv.shutdown()

    def setUp(self):
        REQUESTS.clear()
        self.c = Rostam(self.base, api_key="secret")

    def test_get_with_vector(self):
        p = self.c.get("docs", 1)
        self.assertEqual(p.vector, [1.0, 0.0, 0.0])

    def test_get_without_vector_is_none(self):
        # Normalized to None (not []) when the vector wasn't fetched — must
        # match TcpTransport.get's contract, which already returns None here.
        # See test_get_batch_without_vector_skips_vectors for the get_batch
        # counterpart.
        p = self.c.get("docs", 1, with_vector=False)
        self.assertIsNone(p.vector)

    def test_get_missing_point_returns_none(self):
        # Rostam.get() always appends a query string (?with_vector=...&
        # with_payload=...), so the fake server's "/missing" route match must
        # be done on the parsed path, not the raw request line — otherwise a
        # real 404 never round-trips into a None here.
        self.assertIsNone(self.c.get("docs", "missing"))

    def test_exists_missing_point_is_false(self):
        self.assertFalse(self.c.exists("docs", "missing"))

    def test_health(self):
        # health() is HTTP-only; the facade guards it (raises TransportError
        # on TCP) and forwards it here. Exercise it through the public facade
        # so a broken guard/forward would fail this test too.
        self.assertTrue(self.c.health())

    def test_auth_header_sent(self):
        self.c.health()
        self.assertEqual(REQUESTS[-1]["auth"], "Bearer secret")

    def test_create_collection_body(self):
        self.c.create_collection("docs", dim=3, metric="l2")
        body = REQUESTS[-1]["body"]
        self.assertEqual(body["name"], "docs")
        self.assertEqual(body["config"]["dim"], 3)
        self.assertEqual(body["config"]["metric"], "l2")

    def test_upsert_encodes_metadata(self):
        self.c.upsert("docs", 1, [1.0, 0.0, 0.0], content="hello", metadata={"doc_id": 7, "tag": "x", "ok": True})
        body = REQUESTS[-1]["body"]
        self.assertEqual(body["id"], 1)
        self.assertTrue(body["upsert"])
        self.assertEqual(body["metadata"]["doc_id"], {"kind": "int", "int": 7})
        self.assertEqual(body["metadata"]["tag"], {"kind": "string", "str": "x"})
        self.assertEqual(body["metadata"]["ok"], {"kind": "bool", "bool": True})

    def test_insert_sets_upsert_false(self):
        self.c.insert("docs", 2, [0.0, 1.0, 0.0])
        self.assertFalse(REQUESTS[-1]["body"]["upsert"])

    def test_search(self):
        res = self.c.search("docs", [1.0, 0.0, 0.0], k=3)
        self.assertEqual(len(res), 1)
        self.assertEqual(res[0].id, 1)
        self.assertAlmostEqual(res[0].distance, 0.5)

    def test_search_docs_decodes_metadata(self):
        docs = self.c.search_docs("docs", [1.0, 0.0, 0.0], k=2, filter=f.eq("doc_id", 7))
        self.assertEqual(docs[0].content, "hello")
        self.assertEqual(docs[0].metadata, {"doc_id": 7, "tag": "x"})
        # The filter was forwarded in tagged form.
        sent = REQUESTS[-1]["body"]["filter"]
        self.assertEqual(sent, {"op": "eq", "field": "doc_id", "value": {"kind": "int", "int": 7}})

    def test_search_groups_decodes_key(self):
        groups = self.c.search_groups("docs", [1.0, 0.0, 0.0], k=2, group_by="doc_id")
        self.assertEqual(len(groups), 1)
        self.assertEqual(groups[0].key, 1)
        self.assertEqual(groups[0].hits[0].id, 1)
        self.assertEqual(REQUESTS[-1]["body"]["group_by"], "doc_id")

    def test_hybrid(self):
        res = self.c.hybrid_search("docs", [1.0, 0.0], k=5, sparse={"indices": [3], "values": [0.7]})
        self.assertEqual(res[0].id, 2)
        self.assertEqual(REQUESTS[-1]["body"]["sparse"], {"indices": [3], "values": [0.7]})

    def test_search_text(self):
        # search_text() is HTTP-only (no TCP equivalent); the facade guards
        # and forwards it. Exercise it through the public facade.
        docs = self.c.search_text("docs", "quick fox", k=5)
        self.assertEqual(docs[0].id, 4)
        self.assertEqual(docs[0].content, "the quick brown fox")
        self.assertEqual(docs[0].metadata, {"tag": "y"})
        # The RAW text is forwarded (the SDK ships no tokens).
        self.assertEqual(REQUESTS[-1]["body"]["text"], "quick fox")
        self.assertEqual(REQUESTS[-1]["body"]["k"], 5)

    def test_hybrid_text(self):
        res = self.c.hybrid_text("docs", [1.0, 0.0], "quick fox", k=5, method="rrf")
        self.assertEqual(res[0].id, 4)
        body = REQUESTS[-1]["body"]
        self.assertEqual(body["vector"], [1.0, 0.0])
        self.assertEqual(body["text"], "quick fox")
        self.assertEqual(body["method"], "rrf")

    def test_search_text_global_idf(self):
        # Default omits the flag (byte-identical to the pre-global request body).
        self.c.search_text("docs", "quick fox", k=5)
        self.assertNotIn("global_idf", REQUESTS[-1]["body"])
        # global_idf=True opts into the two-phase global-DF (dfs) path.
        self.c.search_text("docs", "quick fox", k=5, global_idf=True)
        self.assertIs(REQUESTS[-1]["body"]["global_idf"], True)

    def test_hybrid_text_global_idf(self):
        self.c.hybrid_text("docs", [1.0, 0.0], "quick fox", k=5, method="rrf")
        self.assertNotIn("global_idf", REQUESTS[-1]["body"])
        self.c.hybrid_text("docs", [1.0, 0.0], "quick fox", k=5, method="rrf", global_idf=True)
        self.assertIs(REQUESTS[-1]["body"]["global_idf"], True)

    def test_create_collection_full_text(self):
        # full_text=True enables the default analyzer.
        self.c.create_collection("docs", dim=3, full_text=True)
        self.assertEqual(REQUESTS[-1]["body"]["config"]["full_text"], {})
        # A dict tunes the BM25 knobs.
        self.c.create_collection("docs2", dim=3, full_text={"analyzer": "english", "k1": 1.5})
        self.assertEqual(REQUESTS[-1]["body"]["config"]["full_text"], {"analyzer": "english", "k1": 1.5})
        # Omitted => no full_text key (back-compat).
        self.c.create_collection("docs3", dim=3)
        self.assertNotIn("full_text", REQUESTS[-1]["body"]["config"])

    def test_create_collection_sq_prq(self):
        # quant="sq" + sq_bits puts the right fields in the request body.
        self.c.create_collection("sqc", dim=8, quant="sq", sq_bits=6)
        cfg = REQUESTS[-1]["body"]["config"]
        self.assertEqual(cfg["quant"], "sq")
        self.assertEqual(cfg["sq_bits"], 6)
        self.assertNotIn("prq_layers", cfg)
        # quant="prq" + prq_layers.
        self.c.create_collection("prqc", dim=8, quant="prq", prq_layers=3)
        cfg = REQUESTS[-1]["body"]["config"]
        self.assertEqual(cfg["quant"], "prq")
        self.assertEqual(cfg["prq_layers"], 3)
        self.assertNotIn("sq_bits", cfg)
        # Omitted (default 0) => no sq_bits/prq_layers keys (byte-compatible default).
        self.c.create_collection("sqdef", dim=8, quant="sq")
        cfg = REQUESTS[-1]["body"]["config"]
        self.assertEqual(cfg["quant"], "sq")
        self.assertNotIn("sq_bits", cfg)
        self.assertNotIn("prq_layers", cfg)

    def test_create_collection_vamana(self):
        # index_type="vamana" + the R/L/alpha knobs put the right fields in the body.
        self.c.create_collection(
            "vam", dim=8, index_type="vamana", vamana_r=48, vamana_l=80, vamana_alpha=1.3
        )
        cfg = REQUESTS[-1]["body"]["config"]
        self.assertEqual(cfg["index_type"], "vamana")
        self.assertEqual(cfg["vamana_r"], 48)
        self.assertEqual(cfg["vamana_l"], 80)
        self.assertEqual(cfg["vamana_alpha"], 1.3)
        # index_type="vamana" with default R/L/alpha => only index_type, no knobs.
        self.c.create_collection("vamdef", dim=8, index_type="vamana")
        cfg = REQUESTS[-1]["body"]["config"]
        self.assertEqual(cfg["index_type"], "vamana")
        self.assertNotIn("vamana_r", cfg)
        self.assertNotIn("vamana_l", cfg)
        self.assertNotIn("vamana_alpha", cfg)
        # Omitted entirely => no index_type / vamana keys (byte-compatible default).
        self.c.create_collection("plain", dim=8)
        cfg = REQUESTS[-1]["body"]["config"]
        self.assertNotIn("index_type", cfg)
        self.assertNotIn("vamana_r", cfg)
        self.assertNotIn("vamana_l", cfg)
        self.assertNotIn("vamana_alpha", cfg)

    def test_create_collection_scann(self):
        # anisotropic_eta / soar / soar_lambda / pq_nbits put the right fields in
        # the request body.
        self.c.create_collection(
            "scann", dim=8, quant="pq", index_type="ivf",
            anisotropic_eta=4, soar=True, soar_lambda=2, pq_nbits=4,
        )
        cfg = REQUESTS[-1]["body"]["config"]
        self.assertEqual(cfg["anisotropic_eta"], 4)
        self.assertEqual(cfg["soar"], True)
        self.assertEqual(cfg["soar_lambda"], 2)
        self.assertEqual(cfg["pq_nbits"], 4)
        # Omitted (defaults) => no ScaNN keys (byte-compatible default).
        self.c.create_collection("plain2", dim=8)
        cfg = REQUESTS[-1]["body"]["config"]
        self.assertNotIn("anisotropic_eta", cfg)
        self.assertNotIn("soar", cfg)
        self.assertNotIn("soar_lambda", cfg)
        self.assertNotIn("pq_nbits", cfg)

    def test_delete(self):
        self.assertTrue(self.c.delete("docs", 1))
        self.assertFalse(self.c.delete("docs", "missing"))

    def test_delete_by_filter(self):
        # delete_by_filter() is HTTP-only; the facade guards and forwards it.
        n = self.c.delete_by_filter("docs", f.eq("doc_id", 2))
        self.assertEqual(n, 2)

    def test_error_status(self):
        with self.assertRaises(RostamError) as cm:
            self.c.create_collection("x", dim=3, metric="bogus")
        self.assertEqual(cm.exception.status, 400)
        self.assertIn("metric", cm.exception.message)

    def test_filter_builders(self):
        self.assertEqual(f.gte("price", 10.0),
                         {"op": "gte", "field": "price", "value": {"kind": "float", "flt": 10.0}})
        compound = f.and_(f.eq("a", 1), f.in_("b", ["x", "y"]))
        self.assertEqual(compound["op"], "and")
        self.assertEqual(compound["and"][1]["value"], {"kind": "strings", "strs": ["x", "y"]})


def test_get_batch_returns_vectors_content_metadata_and_omits_missing():
    srv = FakeRostam()
    try:
        c = Rostam(srv.url)
        c.create_collection("docs", dim=2, metric="l2")
        c.upsert("docs", 1, [1.0, 0.0], content="hello", metadata={"doc_id": 7})
        c.upsert("docs", 2, [0.0, 1.0], content="world", metadata={"doc_id": 8})

        pts = c.get_batch("docs", [1, 2, 999])
        assert isinstance(pts[0], Point)
        by_id = {p.id: p for p in pts}
        assert set(by_id) == {1, 2}              # 999 missing -> omitted
        assert by_id[1].vector == [1.0, 0.0]
        assert by_id[1].content == "hello"
        assert by_id[1].metadata == {"doc_id": 7}   # $content stripped
        assert "$content" not in by_id[1].metadata
    finally:
        srv.close()


def test_get_batch_without_vector_skips_vectors():
    srv = FakeRostam()
    try:
        c = Rostam(srv.url)
        c.create_collection("docs", dim=2)
        c.upsert("docs", 1, [1.0, 2.0], content="x", metadata={"a": 1})
        pts = c.get_batch("docs", [1], with_vector=False)
        # Normalized to None (not []) — matches TcpTransport.get_batch, so a
        # caller checking "was the vector fetched" behaves the same on both
        # transports. See test_get_without_vector_is_none for the
        # single-point get() counterpart.
        assert pts[0].vector is None
        assert pts[0].content == "x"
    finally:
        srv.close()


def test_scroll_paginates_with_cursor():
    srv = FakeRostam()
    try:
        c = Rostam(srv.url)
        c.create_collection("docs", dim=2, metric="l2")
        for i in range(1, 6):  # ids 1..5
            c.upsert("docs", i, [float(i), 0.0], content=f"c{i}", metadata={"n": i})

        page1 = c.scroll("docs", limit=2)
        assert isinstance(page1, ScrollPage)
        assert [d.id for d in page1] == [1, 2]      # id-ascending, capped at limit
        assert len(page1) == 2                       # lens like a list (haystack relies on this)
        assert page1.next_cursor                     # more data -> non-empty cursor

        page2 = c.scroll("docs", limit=2, cursor=page1.next_cursor)
        assert [d.id for d in page2] == [3, 4]       # strictly after page 1, disjoint
        assert page2.next_cursor

        page3 = c.scroll("docs", limit=2, cursor=page2.next_cursor)
        assert [d.id for d in page3] == [5]
        assert page3.next_cursor == ""               # exhausted -> empty cursor

        # The three pages are disjoint and cover the whole collection in order.
        seen = [d.id for p in (page1, page2, page3) for d in p]
        assert seen == [1, 2, 3, 4, 5]
    finally:
        srv.close()


if __name__ == "__main__":
    unittest.main()
