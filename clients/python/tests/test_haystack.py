"""Haystack adapter tests (skipped without haystack-ai), against the stateful
fake store."""

import unittest

from rostam import Rostam

try:
    from haystack import Document

    HAVE_HS = True
except Exception:  # pragma: no cover
    HAVE_HS = False

from _fakestore import FakeRostam


def _doc(did, text, topic):
    return Document(id=did, content=text, meta={"topic": topic}, embedding=[float(len(text)), 1.0, 0.0])


@unittest.skipUnless(HAVE_HS, "haystack-ai not installed")
class HaystackAdapterTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fake = FakeRostam()

    @classmethod
    def tearDownClass(cls):
        cls.fake.close()

    def setUp(self):
        from rostam.haystack import RostamDocumentStore

        self.fake.docs.clear()  # isolate each test (the fake is shared per class)
        Rostam(self.fake.url).create_collection("hs", dim=3, metric="l2")
        self.store = RostamDocumentStore(url=self.fake.url, collection="hs")

    def test_write_count_filter(self):
        n = self.store.write_documents([_doc("d1", "alpha", "cats"), _doc("d2", "beta", "dogs")])
        self.assertEqual(n, 2)
        self.assertEqual(self.store.count_documents(), 2)
        cats = self.store.filter_documents({"field": "meta.topic", "operator": "==", "value": "cats"})
        self.assertEqual([d.id for d in cats], ["d1"])
        self.assertEqual(cats[0].content, "alpha")
        self.assertEqual(cats[0].meta.get("topic"), "cats")
        # The reserved id key is not surfaced in user meta.
        self.assertNotIn("_hs_id", cats[0].meta)

    def test_embedding_retrieval(self):
        self.store.write_documents([_doc("d1", "alpha", "cats"), _doc("d2", "betabeta", "dogs")])
        hits = self.store._embedding_retrieval([5.0, 1.0, 0.0], top_k=2)
        self.assertEqual(hits[0].id, "d1")  # closest to the query
        self.assertTrue(all(h.score and h.score > 0 for h in hits))

    def test_retriever_component(self):
        from rostam.haystack import RostamEmbeddingRetriever

        self.store.write_documents([_doc("d1", "alpha", "cats")])
        r = RostamEmbeddingRetriever(document_store=self.store, top_k=3)
        out = r.run(query_embedding=[5.0, 1.0, 0.0])
        self.assertEqual([d.id for d in out["documents"]], ["d1"])
        # Serializes (and round-trips the nested store config).
        d = r.to_dict()
        self.assertIn("document_store", d["init_parameters"])
        RostamEmbeddingRetriever.from_dict(d)

    def test_delete(self):
        self.store.write_documents([_doc("d1", "alpha", "cats"), _doc("d2", "beta", "dogs")])
        self.store.delete_documents(["d1"])
        self.assertEqual(self.store.count_documents(), 1)

    def test_requires_embedding(self):
        with self.assertRaises(ValueError):
            self.store.write_documents([Document(id="x", content="no vector")])


if __name__ == "__main__":
    unittest.main()
