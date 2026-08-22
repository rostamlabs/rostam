"""Embedding-helper tests.

Two stdlib HTTP fakes — one mimicking an OpenAI-compatible /embeddings endpoint,
one mimicking the Rostam REST store — so the provider request/response handling
and the TextStore text-first flow are exercised over real sockets with no
third-party deps and no network/API key.
"""

import json
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from rostam import FunctionEmbedder, OpenAIEmbedder, Rostam, TextStore
from _wire import read_body

# ---- fake OpenAI-compatible embeddings endpoint ----

OPENAI_REQS = []


class _OpenAIHandler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(n))
        OPENAI_REQS.append({"auth": self.headers.get("Authorization"), "body": body})
        inputs = body["input"]
        # Return embeddings deterministically (text length feature), and
        # out-of-order to prove the client sorts by index.
        data = [{"index": i, "embedding": [float(len(t)), 1.0]} for i, t in enumerate(inputs)]
        data.reverse()
        payload = json.dumps({"data": data, "model": body["model"]}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


# ---- fake Rostam REST store (stateful) ----

STORE = {}
LAST = {}


class _RostamHandler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _body(self):
        return read_body(self.headers, self.rfile)

    def _send(self, code, obj):
        b = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def do_POST(self):
        body = self._body()
        LAST["body"], LAST["path"] = body, self.path
        p = self.path
        if p == "/v1/collections":
            return self._send(201, {"name": body["name"]})
        if p.endswith("/points"):
            STORE[body["id"]] = {"content": body["content"], "metadata": body.get("metadata") or {}}
            return self._send(200, {"id": body["id"]})
        if p.endswith("/search/docs"):
            docs = [{"id": i, "distance": 0.1, "content": d["content"], "metadata": d["metadata"]}
                    for i, d in STORE.items()][: body["k"]]
            return self._send(200, {"documents": docs})
        self._send(404, {"error": "not found"})

    def do_DELETE(self):
        self._body()
        pid = int(self.path.rsplit("/", 1)[1])
        return self._send(200, {"deleted": STORE.pop(pid, None) is not None})


class OpenAIEmbedderTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.srv = ThreadingHTTPServer(("127.0.0.1", 0), _OpenAIHandler)
        threading.Thread(target=cls.srv.serve_forever, daemon=True).start()
        host, port = cls.srv.server_address
        cls.base = f"http://{host}:{port}/v1"

    @classmethod
    def tearDownClass(cls):
        cls.srv.shutdown()

    def setUp(self):
        OPENAI_REQS.clear()

    def test_embed_documents_order_and_request(self):
        e = OpenAIEmbedder(model="m", api_key="k", base_url=self.base, dimensions=2)
        vecs = e.embed_documents(["aa", "bbbb"])
        # Despite the server returning reversed order, results match input order.
        self.assertEqual(vecs, [[2.0, 1.0], [4.0, 1.0]])
        req = OPENAI_REQS[-1]
        self.assertEqual(req["auth"], "Bearer k")
        self.assertEqual(req["body"]["model"], "m")
        self.assertEqual(req["body"]["input"], ["aa", "bbbb"])
        self.assertEqual(req["body"]["dimensions"], 2)

    def test_embed_query(self):
        e = OpenAIEmbedder(base_url=self.base, api_key="k")
        self.assertEqual(e.embed_query("hello"), [5.0, 1.0])

    def test_empty_documents(self):
        e = OpenAIEmbedder(base_url=self.base)
        self.assertEqual(e.embed_documents([]), [])


class TextStoreTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.srv = ThreadingHTTPServer(("127.0.0.1", 0), _RostamHandler)
        threading.Thread(target=cls.srv.serve_forever, daemon=True).start()
        host, port = cls.srv.server_address
        cls.base = f"http://{host}:{port}"

    @classmethod
    def tearDownClass(cls):
        cls.srv.shutdown()

    def setUp(self):
        STORE.clear()
        LAST.clear()
        embedder = FunctionEmbedder(lambda ts: [[float(len(t)), 1.0, 0.0] for t in ts])
        self.store = TextStore(Rostam(self.base), "docs", embedder)

    def test_create_collection_infers_dim(self):
        self.store.create_collection()
        self.assertEqual(LAST["path"], "/v1/collections")
        self.assertEqual(LAST["body"]["config"]["dim"], 3)  # inferred from the probe

    def test_add_embeds_and_stores(self):
        ids = self.store.add(["alpha", "beta"], metadatas=[{"doc_id": 1}, {"doc_id": 2}], ids=["10", "20"])
        self.assertEqual(ids, ["10", "20"])
        self.assertEqual(STORE[10]["content"], "alpha")
        self.assertEqual(STORE[10]["metadata"]["doc_id"], {"kind": "int", "int": 1})

    def test_add_generates_stable_ids(self):
        a = self.store.add(["same text"])
        b = self.store.add(["same text"])
        self.assertEqual(a, b)  # content-addressed → stable across calls

    def test_search_embeds_query(self):
        self.store.add(["hello world"], metadatas=[{"doc_id": 7}], ids=["1"])
        docs = self.store.search("query", k=3)
        self.assertEqual(docs[0].content, "hello world")
        self.assertEqual(docs[0].metadata["doc_id"], 7)  # decoded to native int

    def test_delete(self):
        self.store.add(["x"], ids=["5"])
        self.assertEqual(self.store.delete(["5"]), 1)
        self.assertNotIn(5, STORE)


if __name__ == "__main__":
    unittest.main()
