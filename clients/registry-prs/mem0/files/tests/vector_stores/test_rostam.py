from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

pytest.importorskip("rostam", reason="rostam-client not installed")

from rostam import RostamError

from mem0.vector_stores.rostam import OutputData, RostamDB, _to_point_id


def _make_point(id, distance=None, vector=None, metadata=None):
    """Helper mirroring rostam's Point/SearchResult shape."""
    return SimpleNamespace(id=id, distance=distance, vector=vector, metadata=metadata or {})


@pytest.fixture
def mock_client():
    with patch("mem0.vector_stores.rostam.RostamClient") as MockClient:
        client_instance = MagicMock()
        MockClient.return_value = client_instance
        yield client_instance


@pytest.fixture
def db(mock_client):
    return RostamDB(
        collection_name="test_col",
        embedding_model_dims=4,
        url="http://localhost:8080",
    )


# ── Initialization ──────────────────────────────────────────────────


class TestInit:
    def test_init_creates_collection(self, mock_client):
        db = RostamDB(
            collection_name="my_col",
            embedding_model_dims=128,
            url="http://localhost:8080",
            api_key="key",
            metric="cosine",
        )
        assert db.collection_name == "my_col"
        assert db.embedding_model_dims == 128
        assert db.metric == "cosine"
        mock_client.create_collection.assert_called_once_with("my_col", 128, metric="cosine")

    def test_init_defaults(self, mock_client):
        db = RostamDB(collection_name="test", embedding_model_dims=4)
        assert db.metric == "cosine"

    def test_init_swallows_already_exists(self, mock_client):
        mock_client.create_collection.side_effect = RostamError("collection already exists")
        # Should not raise.
        RostamDB(collection_name="test", embedding_model_dims=4)

    def test_init_reraises_other_errors(self, mock_client):
        mock_client.create_collection.side_effect = RostamError("connection refused")
        with pytest.raises(RostamError, match="connection refused"):
            RostamDB(collection_name="test", embedding_model_dims=4)


# ── _to_point_id ─────────────────────────────────────────────────────


class TestToPointId:
    def test_numeric_string_used_verbatim(self):
        assert _to_point_id("42") == 42

    def test_non_numeric_string_is_hashed(self):
        v1 = _to_point_id("uuid-like-id")
        v2 = _to_point_id("uuid-like-id")
        v3 = _to_point_id("a-different-id")
        assert v1 == v2
        assert v1 != v3
        assert isinstance(v1, int)


# ── insert ───────────────────────────────────────────────────────────


class TestInsert:
    def test_insert_requires_ids(self, db):
        with pytest.raises(ValueError, match="requires ids"):
            db.insert([[0.1, 0.2, 0.3, 0.4]])

    def test_insert_with_ids_and_payloads(self, db):
        vectors = [[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]]
        payloads = [{"data": "hello"}, {"data": "world"}]
        ids = ["id1", "id2"]

        db.insert(vectors, payloads, ids)

        assert db.client.upsert.call_count == 2
        first_call = db.client.upsert.call_args_list[0]
        assert first_call.args[0] == "test_col"
        assert first_call.args[1] == _to_point_id("id1")
        assert first_call.args[2] == [0.1, 0.2, 0.3, 0.4]
        assert first_call.kwargs["metadata"]["data"] == "hello"
        assert first_call.kwargs["metadata"]["_mem0_id"] == "id1"

    def test_insert_without_payloads(self, db):
        db.insert([[0.1, 0.2, 0.3, 0.4]], ids=["id1"])
        call = db.client.upsert.call_args
        assert call.kwargs["metadata"] == {"_mem0_id": "id1"}


# ── search ───────────────────────────────────────────────────────────


class TestSearch:
    def test_search_basic(self, db):
        db.client.search_docs.return_value = [
            _make_point(1, distance=0.0, metadata={"_mem0_id": "id1", "data": "hello"}),
        ]
        results = db.search("query", [0.1, 0.2, 0.3, 0.4], top_k=2)

        db.client.search_docs.assert_called_once_with("test_col", [0.1, 0.2, 0.3, 0.4], 2, filter=None)
        assert len(results) == 1
        assert results[0].id == "id1"
        assert results[0].score == pytest.approx(1.0)
        assert results[0].payload == {"data": "hello"}

    def test_search_unwraps_nested_query_vector(self, db):
        db.client.search_docs.return_value = []
        db.search("query", [[0.1, 0.2, 0.3, 0.4]], top_k=5)
        call = db.client.search_docs.call_args
        assert call.args[1] == [0.1, 0.2, 0.3, 0.4]

    def test_search_with_equality_filter(self, db):
        db.client.search_docs.return_value = []
        db.search("query", [0.1, 0.2, 0.3, 0.4], filters={"user_id": "alice"})
        call = db.client.search_docs.call_args
        assert call.kwargs["filter"] == {
            "op": "eq",
            "field": "user_id",
            "value": {"kind": "string", "str": "alice"},
        }

    def test_search_with_membership_filter(self, db):
        db.client.search_docs.return_value = []
        db.search("query", [0.1, 0.2, 0.3, 0.4], filters={"user_id": ["alice", "bob"]})
        call = db.client.search_docs.call_args
        assert call.kwargs["filter"]["op"] == "in"

    def test_search_score_conversion(self, db):
        db.client.search_docs.return_value = [_make_point(1, distance=1.0, metadata={})]
        results = db.search("query", [0.0, 0.0, 0.0, 0.0])
        assert results[0].score == pytest.approx(0.5)


# ── delete ───────────────────────────────────────────────────────────


class TestDelete:
    def test_delete(self, db):
        db.delete("id1")
        db.client.delete.assert_called_once_with("test_col", _to_point_id("id1"))


# ── update ───────────────────────────────────────────────────────────


class TestUpdate:
    def test_update_with_vector_and_payload(self, db):
        db.update("id1", vector=[0.5, 0.6, 0.7, 0.8], payload={"data": "updated"})
        db.client.upsert.assert_called_once_with(
            "test_col",
            _to_point_id("id1"),
            [0.5, 0.6, 0.7, 0.8],
            metadata={"data": "updated", "_mem0_id": "id1"},
        )
        db.client.get_batch.assert_not_called()

    def test_update_vector_only_reads_existing_payload(self, db):
        db.client.get_batch.return_value = [
            _make_point(_to_point_id("id1"), vector=[0.1] * 4, metadata={"_mem0_id": "id1", "data": "old"})
        ]
        db.update("id1", vector=[0.9] * 4)

        db.client.get_batch.assert_called_once_with("test_col", [_to_point_id("id1")])
        call = db.client.upsert.call_args
        assert call.args[2] == [0.9] * 4
        assert call.kwargs["metadata"]["data"] == "old"

    def test_update_payload_only_reads_existing_vector(self, db):
        db.client.get_batch.return_value = [
            _make_point(_to_point_id("id1"), vector=[0.2] * 4, metadata={"_mem0_id": "id1"})
        ]
        db.update("id1", payload={"data": "new"})

        call = db.client.upsert.call_args
        assert call.args[2] == [0.2] * 4
        assert call.kwargs["metadata"]["data"] == "new"

    def test_update_not_found_raises(self, db):
        db.client.get_batch.return_value = []
        with pytest.raises(ValueError, match="not found"):
            db.update("missing")


# ── get ──────────────────────────────────────────────────────────────


class TestGet:
    def test_get_found(self, db):
        db.client.get_batch.return_value = [
            _make_point(_to_point_id("id1"), metadata={"_mem0_id": "id1", "data": "hello"})
        ]
        result = db.get("id1")

        db.client.get_batch.assert_called_once_with("test_col", [_to_point_id("id1")], with_vector=False)
        assert result is not None
        assert result.id == "id1"
        assert result.payload == {"data": "hello"}

    def test_get_not_found(self, db):
        db.client.get_batch.return_value = []
        assert db.get("missing") is None


# ── list_cols ────────────────────────────────────────────────────────


class TestListCols:
    def test_list_cols_not_implemented(self, db):
        with pytest.raises(NotImplementedError):
            db.list_cols()


# ── delete_col ───────────────────────────────────────────────────────


class TestDeleteCol:
    def test_delete_col(self, db):
        db.delete_col()
        db.client.drop_collection.assert_called_once_with("test_col")


# ── col_info ─────────────────────────────────────────────────────────


class TestColInfo:
    def test_col_info(self, db):
        info = db.col_info()
        assert info == {"name": "test_col", "dimension": 4, "distance": "cosine"}


# ── list ─────────────────────────────────────────────────────────────


class TestList:
    def test_list_returns_tuple(self, db):
        db.client.scroll.return_value = [
            _make_point(1, metadata={"_mem0_id": "id1", "data": "hello"}),
            _make_point(2, metadata={"_mem0_id": "id2", "data": "world"}),
        ]
        results, cursor = db.list()

        assert cursor is None
        assert len(results) == 2
        assert results[0].id == "id1"
        assert results[0].score is None

    def test_list_with_filters_and_top_k(self, db):
        db.client.scroll.return_value = []
        db.list(filters={"user_id": "alice"}, top_k=50)
        call = db.client.scroll.call_args
        assert call.kwargs["limit"] == 50
        assert call.kwargs["filter"] == {
            "op": "eq",
            "field": "user_id",
            "value": {"kind": "string", "str": "alice"},
        }

    def test_list_default_limit(self, db):
        db.client.scroll.return_value = []
        db.list()
        call = db.client.scroll.call_args
        assert call.kwargs["limit"] == 0


# ── reset ────────────────────────────────────────────────────────────


class TestReset:
    def test_reset_drops_and_recreates(self, db):
        db.reset()
        db.client.drop_collection.assert_called_once_with("test_col")
        # create_collection called once at init, once again on reset.
        assert db.client.create_collection.call_count == 2


# ── Config ───────────────────────────────────────────────────────────


class TestConfig:
    def test_config_defaults(self):
        from mem0.configs.vector_stores.rostam import RostamConfig

        config = RostamConfig()
        assert config.collection_name == "mem0"
        assert config.embedding_model_dims == 1536
        assert config.url == "http://localhost:8080"
        assert config.metric == "cosine"

    def test_config_custom_values(self):
        from mem0.configs.vector_stores.rostam import RostamConfig

        config = RostamConfig(
            collection_name="custom",
            embedding_model_dims=768,
            url="https://rostam.example.com",
            api_key="rostam-key",
            metric="dot",
        )
        assert config.collection_name == "custom"
        assert config.embedding_model_dims == 768
        assert config.url == "https://rostam.example.com"
        assert config.api_key == "rostam-key"
        assert config.metric == "dot"

    def test_config_rejects_extra_fields(self):
        from mem0.configs.vector_stores.rostam import RostamConfig

        with pytest.raises(ValueError, match="Extra fields not allowed"):
            RostamConfig(unknown_field="value")


# ── Factory Registration ─────────────────────────────────────────────


class TestFactoryRegistration:
    def test_vector_store_factory_has_rostam(self):
        from mem0.utils.factory import VectorStoreFactory

        assert "rostam" in VectorStoreFactory.provider_to_class
        assert VectorStoreFactory.provider_to_class["rostam"] == "mem0.vector_stores.rostam.RostamDB"

    def test_vector_store_config_has_rostam(self):
        from mem0.vector_stores.configs import VectorStoreConfig

        config = VectorStoreConfig.__private_attributes__["_provider_configs"].default
        assert "rostam" in config
        assert config["rostam"] == "RostamConfig"

    def test_config_validation_pipeline(self, mock_client):
        from mem0.vector_stores.configs import VectorStoreConfig

        config = VectorStoreConfig(
            provider="rostam",
            config={"collection_name": "test", "url": "http://localhost:8080"},
        )
        assert config.config.collection_name == "test"
        assert config.config.url == "http://localhost:8080"
        assert config.config.embedding_model_dims == 1536


# ── OutputData ───────────────────────────────────────────────────────


class TestOutputData:
    def test_output_data_has_required_fields(self):
        od = OutputData(id="test", score=0.9, payload={"data": "hello"})
        assert od.id == "test"
        assert od.score == 0.9
        assert od.payload == {"data": "hello"}
