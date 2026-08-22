"""Multi-vector (late-interaction / MaxSim) client tests against a stdlib fake.

The fake stores added documents and computes a real (un-normalized) MaxSim so
search ordering is meaningful, letting the test verify the client's token-matrix
encoding, metadata encode/decode, and result parsing over a real socket.
"""

import json
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from rostam import Rostam
from _wire import read_body

DOCS = {}   # id -> {"tokens": [[...]], "metadata": tagged}
LAST = {}


def _maxsim(query, doc_tokens):
    score = 0.0
    for q in query:
        best = None
        for d in doc_tokens:
            s = sum(a * b for a, b in zip(q, d))
            best = s if best is None or s > best else best
        score += best or 0.0
    return score


class _Handler(BaseHTTPRequestHandler):
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
        if p == "/v1/multivector/docs":
            return self._send(201, {"name": "docs"})
        if p == "/v1/multivector/docs/docs":
            DOCS[body["id"]] = {"tokens": body["tokens"], "metadata": body.get("metadata") or {}}
            return self._send(200, {"id": body["id"]})
        if p == "/v1/multivector/docs/search":
            scored = [{"id": i, "score": _maxsim(body["query"], d["tokens"]), "metadata": d["metadata"]}
                      for i, d in DOCS.items()]
            scored.sort(key=lambda r: r["score"], reverse=True)
            return self._send(200, {"results": scored[: body["k"]]})
        self._send(404, {"error": "not found"})

    def do_DELETE(self):
        self._body()
        pid = int(self.path.rsplit("/", 1)[1])
        return self._send(200, {"deleted": DOCS.pop(pid, None) is not None})


class MultiVectorClientTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.srv = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
        threading.Thread(target=cls.srv.serve_forever, daemon=True).start()
        host, port = cls.srv.server_address
        cls.base = f"http://{host}:{port}"

    @classmethod
    def tearDownClass(cls):
        cls.srv.shutdown()

    def setUp(self):
        DOCS.clear()
        LAST.clear()
        self.c = Rostam(self.base)

    def test_create_sends_config(self):
        self.c.mv_create_collection("docs", dim=4, m=8, quant="sq8", persistent=True)
        b = LAST["body"]
        self.assertEqual((b["dim"], b["m"], b["quant"], b["persistent"]), (4, 8, "sq8", True))

    def test_add_encodes_tokens_and_metadata(self):
        self.c.mv_add("docs", 1, [[1.0, 0.0], [0.0, 1.0]], metadata={"doc": 7})
        self.assertEqual(LAST["body"]["id"], 1)
        self.assertEqual(LAST["body"]["tokens"], [[1.0, 0.0], [0.0, 1.0]])
        self.assertEqual(LAST["body"]["metadata"]["doc"], {"kind": "int", "int": 7})

    def test_search_ranks_and_decodes(self):
        self.c.mv_add("docs", 1, [[1.0, 0.0], [0.0, 1.0]], metadata={"doc": 1})
        self.c.mv_add("docs", 2, [[0.0, 1.0]], metadata={"doc": 2})
        res = self.c.mv_search("docs", [[1.0, 0.0]], k=2)
        self.assertEqual([r.id for r in res], [1, 2])     # doc 1 aligns with the query
        self.assertEqual(res[0].metadata["doc"], 1)        # decoded to native int
        self.assertGreater(res[0].score, res[1].score)

    def test_delete(self):
        self.c.mv_add("docs", 1, [[1.0, 0.0]])
        self.assertTrue(self.c.mv_delete("docs", 1))
        self.assertNotIn(1, DOCS)


if __name__ == "__main__":
    unittest.main()
