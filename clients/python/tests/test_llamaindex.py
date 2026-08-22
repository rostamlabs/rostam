"""LlamaIndex adapter tests (skipped without llama-index-core), against the
stateful fake store."""

import unittest

import pytest

from rostam import Rostam

try:
    from llama_index.core.schema import NodeRelationship, RelatedNodeInfo, TextNode
    from llama_index.core.vector_stores.types import (
        FilterCondition,
        MetadataFilter,
        MetadataFilters,
        VectorStoreQuery,
    )

    HAVE_LI = True
except Exception:  # pragma: no cover
    HAVE_LI = False

from _fakestore import FakeRostam


@pytest.fixture()
def li_env():
    """Start a FakeRostam server and return (url, make_node(text, embedding))."""
    if not HAVE_LI:
        pytest.skip("llama-index-core not installed")
    fake = FakeRostam()
    def make_node(text, embedding):
        return TextNode(text=text, embedding=embedding)
    yield fake.url, make_node
    fake.close()


@pytest.mark.skipif(not HAVE_LI, reason="llama-index-core not installed")
def test_llamaindex_auto_create_on_first_add(li_env):
    from rostam.llamaindex import RostamVectorStore
    url, make_node = li_env
    client = Rostam(url)
    calls = []
    orig = client.create_collection
    def spy(name, dim, **kw):
        calls.append((name, dim, kw))
        return orig(name, dim, **kw)
    client.create_collection = spy
    store = RostamVectorStore(client=client, collection="li_auto")  # not pre-created
    ids = store.add([make_node("hello world", [1.0, 0.0])])
    assert len(ids) == 1
    # auto-create fired exactly once, with the dim inferred from the embedding
    assert len(calls) == 1, f"expected 1 create_collection call, got {len(calls)}"
    assert calls[0][0] == "li_auto"
    assert calls[0][1] == 2  # len([1.0, 0.0])
    assert calls[0][2].get("metric") == "cosine"
    # second add must NOT call create_collection again
    store.add([make_node("second", [0.5, 0.5])])
    assert len(calls) == 1, "create_collection called more than once"


def _node(text, nid, topic, ref=None):
    n = TextNode(text=text, id_=nid, metadata={"topic": topic})
    n.embedding = [float(len(text)), 1.0, 0.0]
    if ref:
        n.relationships[NodeRelationship.SOURCE] = RelatedNodeInfo(node_id=ref)
    return n


@unittest.skipUnless(HAVE_LI, "llama-index-core not installed")
class LlamaIndexAdapterTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fake = FakeRostam()

    @classmethod
    def tearDownClass(cls):
        cls.fake.close()

    def setUp(self):
        from rostam.llamaindex import RostamVectorStore

        self.fake.docs.clear()  # isolate each test (the fake is shared per class)
        client = Rostam(self.fake.url)
        client.create_collection("li", dim=3, metric="l2")
        self.store = RostamVectorStore(client=client, collection="li")
        self.col = "li"

    def test_add_query_roundtrip(self):
        ids = self.store.add([_node("alpha", "n1", "cats", "docA"), _node("beta", "n2", "dogs", "docB")])
        self.assertEqual(set(ids), {"n1", "n2"})
        res = self.store.query(VectorStoreQuery(query_embedding=[5.0, 1.0, 0.0], similarity_top_k=2))
        # Node reconstructed with id, text, and user metadata.
        node_ids = [n.node_id for n in res.nodes]
        self.assertIn("n1", node_ids)
        top = res.nodes[node_ids.index("n1")]
        self.assertEqual(top.get_content(), "alpha")
        self.assertEqual(top.metadata.get("topic"), "cats")
        self.assertTrue(all(s >= 0 for s in res.similarities))

    def test_delete_by_ref_doc(self):
        self.store.add([_node("alpha", "n1", "cats", "docA"), _node("beta", "n2", "dogs", "docB")])
        self.store.delete("docA")
        res = self.store.query(VectorStoreQuery(query_embedding=[1.0, 1.0, 0.0], similarity_top_k=5))
        self.assertNotIn("n1", [n.node_id for n in res.nodes])
        self.assertIn("n2", [n.node_id for n in res.nodes])

    def test_metadata_filter(self):
        self.store.add([_node("alpha", "n1", "cats", "docA"), _node("beta", "n2", "dogs", "docB")])
        flt = MetadataFilters(filters=[MetadataFilter(key="topic", value="dogs")], condition=FilterCondition.AND)
        res = self.store.query(VectorStoreQuery(query_embedding=[1.0, 1.0, 0.0], similarity_top_k=5, filters=flt))
        self.assertEqual([n.node_id for n in res.nodes], ["n2"])


@pytest.fixture()
def li_hybrid_env():
    """FakeRostam with a full_text collection seeded with three nodes."""
    if not HAVE_LI:
        pytest.skip("llama-index-core not installed")
    fake = FakeRostam()
    url = fake.url
    client = Rostam(url)
    client.create_collection("li_hybrid", dim=2, metric="cosine", full_text=True)

    from rostam.llamaindex import RostamVectorStore
    store = RostamVectorStore(client=client, collection="li_hybrid", auto_create=False, full_text=True)
    nodes = [
        TextNode(text="red apple pie", id_="n1", embedding=[1.0, 0.0]),
        TextNode(text="green apple", id_="n2", embedding=[0.9, 0.1]),
        TextNode(text="blue sky", id_="n3", embedding=[0.0, 1.0]),
    ]
    store.add(nodes)
    yield store
    fake.close()


def test_llamaindex_query_hybrid_mode(li_hybrid_env):
    from llama_index.core.vector_stores.types import VectorStoreQuery, VectorStoreQueryMode
    store = li_hybrid_env  # full_text store, nodes added
    q = VectorStoreQuery(
        query_embedding=[0.0, 0.0],
        query_str="apple pie",
        similarity_top_k=2,
        mode=VectorStoreQueryMode.HYBRID,
    )
    res = store.query(q)
    texts = [n.get_content() for n in res.nodes]
    assert len(res.nodes) >= 1
    assert "red apple pie" in texts[0]   # BM25 lane surfaces the lexical match


@pytest.mark.skipif(not HAVE_LI, reason="llama-index-core not installed")
def test_llamaindex_async_methods(li_env, monkeypatch):
    """Verify async_add/aquery/adelete are explicit overrides routing through asyncio.to_thread."""
    import asyncio
    import rostam.llamaindex as limod
    from rostam.llamaindex import RostamVectorStore

    # (b) Assert the three methods are explicitly defined on RostamVectorStore, not inherited.
    for name in ("async_add", "aquery", "adelete"):
        assert name in RostamVectorStore.__dict__, f"{name} not an explicit override on RostamVectorStore"

    url, make_node = li_env
    store = RostamVectorStore(client=Rostam(url), collection="li_async")

    # (a) Spy on asyncio.to_thread as seen by the module to confirm routing.
    calls = []
    real_to_thread = asyncio.to_thread

    async def spy_to_thread(fn, *args, **kwargs):
        calls.append(fn.__name__)
        return await real_to_thread(fn, *args, **kwargs)

    monkeypatch.setattr(limod.asyncio, "to_thread", spy_to_thread)

    async def run():
        await store.async_add([make_node("hello", [1.0, 0.0])])
        q = VectorStoreQuery(query_embedding=[1.0, 0.0], similarity_top_k=1)
        res = await store.aquery(q)
        await store.adelete("nonexistent-ref")
        return res

    res = asyncio.run(run())

    # All three async methods must have gone through asyncio.to_thread.
    assert "add" in calls, f"async_add did not route through asyncio.to_thread; calls={calls}"
    assert "query" in calls, f"aquery did not route through asyncio.to_thread; calls={calls}"
    assert "delete" in calls, f"adelete did not route through asyncio.to_thread; calls={calls}"

    # Sanity: the aquery result contains the node we added.
    assert res.nodes and res.nodes[0].get_content() == "hello"


@pytest.mark.skipif(not HAVE_LI, reason="llama-index-core not installed")
def test_llamaindex_dict_full_text_preserved(li_env):
    """When full_text is a dict (BM25 tuning), _ensure_collection must pass
    the dict through unchanged — not collapse it to True."""
    from rostam.llamaindex import RostamVectorStore

    url, make_node = li_env
    client = Rostam(url)
    calls = []
    orig = client.create_collection
    def spy(name, dim, **kw):
        calls.append((name, dim, kw))
        return orig(name, dim, **kw)
    client.create_collection = spy

    ft_dict = {"analyzer": "english", "k1": 1.2, "b": 0.75}
    store = RostamVectorStore(client=client, collection="ftdict", full_text=ft_dict)
    store.add([make_node("hello", [1.0, 0.0])])

    assert len(calls) == 1, f"expected 1 create_collection call, got {len(calls)}"
    recorded = calls[0][2].get("full_text")
    assert recorded == ft_dict, (
        f"full_text dict must be preserved by _ensure_collection, got {recorded!r}"
    )


if __name__ == "__main__":
    unittest.main()
