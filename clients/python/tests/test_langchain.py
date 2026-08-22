"""LangChain adapter tests.

Skipped automatically when ``langchain-core`` is not installed. Runs against a
small stateful HTTP fake that stores upserted points and serves them back, so the
adapter's embed → upsert → search → map-to-Document round trip is exercised
end-to-end (including metadata encode/decode and filter translation).
"""

import json
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
from tests._fakestore import FakeRostam

try:
    from langchain_core.embeddings import Embeddings

    HAVE_LC = True
except Exception:  # pragma: no cover - exercised only without langchain
    HAVE_LC = False

from rostam import Rostam
from _wire import read_body

STORE = {}        # id -> {"content":..., "metadata": tagged}
LAST = {"body": None, "path": None}


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
        if p == "/v1/collections":
            return self._send(201, {"name": body["name"]})
        if p.endswith("/points"):
            STORE[body["id"]] = {"content": body["content"], "metadata": body.get("metadata") or {}}
            return self._send(200, {"id": body["id"]})
        if p.endswith("/search/docs"):
            docs = [{"id": i, "distance": 0.1 * n, "content": d["content"], "metadata": d["metadata"]}
                    for n, (i, d) in enumerate(STORE.items())][: body["k"]]
            return self._send(200, {"documents": docs})
        if p.endswith("/search/groups"):
            gb = body["group_by"]
            groups = {}
            for i, d in STORE.items():
                key = d["metadata"].get(gb)
                kk = json.dumps(key)
                groups.setdefault(kk, (key, []))[1].append(
                    {"id": i, "distance": 0.1, "content": d["content"], "metadata": d["metadata"]})
            out = [{"key": key, "hits": hits[: body["group_size"]]} for key, hits in groups.values()][: body["k"]]
            return self._send(200, {"groups": out})
        self._send(404, {"error": "not found"})

    def do_DELETE(self):
        self._body()
        if "/points/" in self.path:
            pid = int(self.path.rsplit("/", 1)[1])
            existed = STORE.pop(pid, None) is not None
            return self._send(200, {"deleted": existed})
        self._send(200, {"dropped": True})


if HAVE_LC:

    class FakeEmbeddings(Embeddings):
        """Deterministic 3-dim embeddings: text length feature, no model needed."""

        def embed_documents(self, texts):
            return [[float(len(t)), 1.0, 0.0] for t in texts]

        def embed_query(self, text):
            return [float(len(text)), 1.0, 0.0]


@unittest.skipUnless(HAVE_LC, "langchain-core not installed")
class LangChainAdapterTest(unittest.TestCase):
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
        STORE.clear()
        from rostam.langchain import RostamVectorStore

        self.store = RostamVectorStore(Rostam(self.base), "docs", FakeEmbeddings())

    def test_add_texts_and_search(self):
        ids = self.store.add_texts(
            ["alpha", "beta"],
            metadatas=[{"doc_id": 1}, {"doc_id": 2}],
            ids=["10", "20"],
        )
        self.assertEqual(ids, ["10", "20"])
        # Numeric ids passed through verbatim; content + tagged metadata stored.
        self.assertIn(10, STORE)
        self.assertEqual(STORE[10]["content"], "alpha")
        self.assertEqual(STORE[10]["metadata"]["doc_id"], {"kind": "int", "int": 1})

        docs = self.store.similarity_search("query", k=2)
        self.assertEqual({d.page_content for d in docs}, {"alpha", "beta"})
        # Metadata decoded back to native ints.
        self.assertEqual(docs[0].metadata["doc_id"], 1)

    def test_similarity_search_with_score_and_filter(self):
        self.store.add_texts(["x"], metadatas=[{"doc_id": 7}], ids=["1"])
        pairs = self.store.similarity_search_with_score("q", k=1, filter={"doc_id": 7})
        self.assertEqual(len(pairs), 1)
        doc, score = pairs[0]
        self.assertEqual(doc.page_content, "x")
        self.assertIsInstance(score, float)
        # The simple {field: value} filter was translated to a tagged eq.
        self.assertEqual(LAST["body"]["filter"],
                         {"op": "eq", "field": "doc_id", "value": {"kind": "int", "int": 7}})

    def test_search_grouped(self):
        self.store.add_texts(
            ["a1", "a2", "b1"],
            metadatas=[{"doc_id": 1}, {"doc_id": 1}, {"doc_id": 2}],
            ids=["1", "2", "3"],
        )
        groups = self.store.search_grouped("q", k=2, group_by="doc_id", group_size=2)
        self.assertEqual(len(groups), 2)
        self.assertTrue(all(isinstance(g, list) for g in groups))

    def test_non_numeric_ids_are_stable(self):
        from rostam.langchain import _to_id

        self.assertEqual(_to_id("123"), 123)
        self.assertEqual(_to_id("doc-abc"), _to_id("doc-abc"))
        self.assertNotEqual(_to_id("doc-abc"), _to_id("doc-xyz"))

    def test_delete(self):
        self.store.add_texts(["a"], ids=["5"])
        self.assertTrue(self.store.delete(["5"]))
        self.assertNotIn(5, STORE)

    def test_from_texts(self):
        from rostam.langchain import RostamVectorStore

        store = RostamVectorStore.from_texts(
            ["hello"], FakeEmbeddings(), metadatas=[{"k": "v"}],
            client=Rostam(self.base), collection="docs", ids=["1"],
        )
        self.assertEqual(STORE[1]["content"], "hello")
        self.assertEqual(store.embeddings.__class__.__name__, "FakeEmbeddings")


if __name__ == "__main__":
    unittest.main()


# ---- pytest-style tests using FakeRostam ----


@pytest.fixture()
def lc_env():
    """Start a FakeRostam server and return (url, FakeEmbeddings())."""
    fake = FakeRostam()
    yield fake.url, FakeEmbeddings()
    fake.close()


@pytest.mark.skipif(not HAVE_LC, reason="langchain-core not installed")
def test_auto_create_on_first_add(lc_env):
    from rostam import Rostam
    from rostam.langchain import RostamVectorStore
    url, emb = lc_env
    client = Rostam(url)
    calls = []
    orig = client.create_collection
    def spy(name, dim, **kw):
        calls.append((name, dim, kw))
        return orig(name, dim, **kw)
    client.create_collection = spy
    store = RostamVectorStore(client, "auto", emb)  # collection NOT pre-created
    store.add_texts(["alpha", "beta"], [{"k": 1}, {"k": 2}])
    # auto-create fired exactly once, with the dim inferred from the embedding
    assert len(calls) == 1, f"expected 1 create_collection call, got {len(calls)}"
    assert calls[0][0] == "auto"
    assert calls[0][1] == len(emb.embed_query("alpha"))
    # second add_texts must NOT call create_collection again
    store.add_texts(["gamma"])
    assert len(calls) == 1, "create_collection called more than once"
    # data path still works end-to-end
    got = store.similarity_search("alpha", k=1)
    assert got and got[0].page_content == "alpha"


@pytest.mark.skipif(not HAVE_LC, reason="langchain-core not installed")
def test_get_by_ids_roundtrips_string_ids(lc_env):
    from rostam import Rostam
    from rostam.langchain import RostamVectorStore
    url, emb = lc_env
    store = RostamVectorStore(Rostam(url), "byid", emb)
    store.add_texts(["one", "two"], [{"n": 1}, {"n": 2}], ids=["id-one", "id-two"])
    docs = store.get_by_ids(["id-one", "id-two", "nope"])
    by_content = {d.page_content: d for d in docs}
    assert set(by_content) == {"one", "two"}        # missing id omitted
    assert by_content["one"].id == "id-one"          # ORIGINAL string id preserved
    assert by_content["one"].metadata == {"n": 1}


@pytest.fixture()
def mmr_env():
    """FakeRostam server + deterministic embeddings stub for MMR tests.

    Texts map to vectors:
      "near1"=[1,0,0]  "near2"=[0.99,0.01,0]  "near3"=[0.98,0.02,0]  "far"=[0,1,0]
    All other texts fall back to [0,0,1] to stay out of the way.
    """
    if not HAVE_LC:
        pytest.skip("langchain-core not installed")

    class MMREmbeddings(Embeddings):
        _MAP = {
            "near1": [1.0, 0.0, 0.0],
            "near2": [0.99, 0.01, 0.0],
            "near3": [0.98, 0.02, 0.0],
            "far":   [0.0, 1.0, 0.0],
        }

        def _vec(self, text):
            return self._MAP.get(text, [0.0, 0.0, 1.0])

        def embed_documents(self, texts):
            return [self._vec(t) for t in texts]

        def embed_query(self, text):
            return self._vec(text)

    from rostam import Rostam
    from rostam.langchain import RostamVectorStore

    fake = FakeRostam()
    store = RostamVectorStore(Rostam(fake.url), "mmr", MMREmbeddings(), metric="cosine")
    yield store
    fake.close()


@pytest.mark.skipif(not HAVE_LC, reason="langchain-core not installed")
def test_mmr_selects_diverse_results(mmr_env):
    # mmr_env: store backed by an embeddings stub where embed_query/embed_documents
    # return the literal vectors below by text. Three near-duplicates cluster near
    # the query; one is far. MMR(k=2, lambda=0.3) should pick 1 from the cluster + the far one.
    store = mmr_env
    # texts -> vectors:
    #   "near1"=[1,0,0] "near2"=[0.99,0.01,0] "near3"=[0.98,0.02,0] "far"=[0,1,0]
    store.add_texts(["near1", "near2", "near3", "far"])
    picked = store.max_marginal_relevance_search("near1", k=2, fetch_k=4, lambda_mult=0.3)
    contents = {d.page_content for d in picked}
    assert "near1" in contents
    assert "far" in contents       # diversity term pulls in the far doc over near2/near3


@pytest.mark.skipif(not HAVE_LC, reason="langchain-core not installed")
def test_from_texts_forwards_options(lc_env):
    from rostam import Rostam
    from rostam.langchain import RostamVectorStore
    url, emb = lc_env
    client = Rostam(url)
    calls = []
    orig = client.create_collection
    def spy(name, dim, **kw):
        calls.append((name, dim, kw))
        return orig(name, dim, **kw)
    client.create_collection = spy
    RostamVectorStore.from_texts(
        ["a", "b"], emb, client=client, collection="ft", metric="l2", full_text=True
    )
    assert len(calls) == 1, f"expected 1 create_collection call, got {len(calls)}"
    assert calls[0][0] == "ft"
    assert calls[0][2].get("metric") == "l2"
    assert calls[0][2].get("full_text")  # truthy (True, not None)


@pytest.mark.skipif(not HAVE_LC, reason="langchain-core not installed")
def test_from_texts_forwards_sparse_embedding(lc_env):
    """sparse_embedding kwarg must propagate into the constructed store's
    _sparse_embedding attribute; previously the forwarding tuple omitted it."""
    from rostam import Rostam
    from rostam.langchain import RostamVectorStore
    url, emb = lc_env
    client = Rostam(url)
    sentinel = lambda q: {"indices": [0], "values": [1.0]}
    store = RostamVectorStore.from_texts(
        ["a", "b"], emb, client=client, collection="sparse_fwd",
        sparse_embedding=sentinel,
    )
    assert store._sparse_embedding is sentinel, (
        "sparse_embedding must be forwarded by from_texts"
    )


@pytest.fixture()
def hybrid_env():
    """FakeRostam server + deterministic embeddings stub for hybrid search tests.

    Uses a full_text=True collection so auto-create enables BM25. The fake fuses
    dense (1/(1+L2)) + term-overlap BM25 additively, so a lexically-matching doc
    outranks a purely-near one for the right query.
    """
    if not HAVE_LC:
        pytest.skip("langchain-core not installed")

    from rostam import Rostam
    from rostam.langchain import RostamVectorStore

    fake = FakeRostam()
    store = RostamVectorStore(Rostam(fake.url), "hybrid", FakeEmbeddings(), full_text=True)
    yield store
    fake.close()


@pytest.mark.skipif(not HAVE_LC, reason="langchain-core not installed")
def test_async_methods_are_explicit_overrides():
    """All six async methods must be defined on RostamVectorStore itself, not
    merely inherited from VectorStore (which uses run_in_executor, not to_thread)."""
    from rostam.langchain import RostamVectorStore
    for name in (
        "aadd_texts", "asimilarity_search", "asimilarity_search_with_score",
        "amax_marginal_relevance_search", "aget_by_ids", "adelete",
    ):
        assert name in RostamVectorStore.__dict__, f"{name} should be an explicit override"


@pytest.mark.skipif(not HAVE_LC, reason="langchain-core not installed")
def test_async_methods_use_to_thread(lc_env):
    """Async overrides must delegate through asyncio.to_thread (not the base
    class's run_in_executor path)."""
    import asyncio
    from unittest.mock import patch
    import rostam.langchain as lcmod
    from rostam import Rostam
    from rostam.langchain import RostamVectorStore
    url, emb = lc_env
    store = RostamVectorStore(Rostam(url), "async2", emb)

    calls = []
    real_to_thread = asyncio.to_thread

    async def spy(fn, *a, **kw):
        calls.append(fn.__name__)
        return await real_to_thread(fn, *a, **kw)

    async def run():
        with patch.object(lcmod.asyncio, "to_thread", spy):
            await store.aadd_texts(["x", "y"], [{"i": 1}, {"i": 2}])
            got = await store.asimilarity_search("x", k=1)
            ids = await store.aget_by_ids([])
        return got, ids

    got, ids = asyncio.run(run())
    assert got and got[0].page_content == "x"
    assert ids == []
    assert "add_texts" in calls, f"aadd_texts did not call to_thread; calls={calls}"
    assert "similarity_search" in calls, f"asimilarity_search did not call to_thread; calls={calls}"
    assert "get_by_ids" in calls, f"aget_by_ids did not call to_thread; calls={calls}"


@pytest.mark.skipif(not HAVE_LC, reason="langchain-core not installed")
def test_hybrid_search_uses_text_fusion(hybrid_env):
    # hybrid_env: store on a full_text collection; fake fuses dense L2 with a simple
    # term-overlap BM25 so a lexically-matching doc outranks a purely-near one.
    store = hybrid_env
    store.add_texts(
        ["red apple pie", "green apple", "blue sky"],
        [{"i": 1}, {"i": 2}, {"i": 3}],
    )
    docs = store.hybrid_search("apple pie", k=2)
    assert docs[0].page_content == "red apple pie"   # lexical match wins via BM25 lane
    assert all(isinstance(d.page_content, str) and d.page_content for d in docs)  # content enriched
