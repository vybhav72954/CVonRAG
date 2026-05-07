"""
tests/test_vector_store.py
Unit tests for app/vector_store.py
Mocking Qdrant and Ollama HTTP endpoints.
"""

import pytest
from unittest.mock import AsyncMock, patch, MagicMock
import httpx
from qdrant_client.http.models import Distance, VectorParams
from qdrant_client import AsyncQdrantClient

from app.vector_store import (
    get_qdrant,
    get_http,
    close_clients,
    embed_text,
    embed_texts,
    ensure_collection_exists,
    collection_info,
    ingest_gold_standard_bullets,
    retrieve_style_exemplars,
)
from app.models import RoleType

@pytest.fixture(autouse=True)
async def cleanup_clients():
    """Ensure clients are closed after each test to keep state clean."""
    yield
    await close_clients()

class TestSingletonClients:
    @pytest.mark.asyncio
    async def test_get_qdrant_creates_singleton(self):
        client1 = get_qdrant()
        client2 = get_qdrant()
        assert client1 is client2
        assert isinstance(client1, AsyncQdrantClient)

    @pytest.mark.asyncio
    async def test_get_http_creates_singleton(self):
        http1 = get_http()
        http2 = get_http()
        assert http1 is http2
        assert isinstance(http1, httpx.AsyncClient)

    @pytest.mark.asyncio
    async def test_close_clients_clears_instances(self):
        get_qdrant()
        get_http()
        await close_clients()
        # Next call creates a new instance
        assert get_qdrant() is not None
        assert get_http() is not None

class TestEmbeddings:
    @pytest.mark.asyncio
    async def test_embed_text_success(self):
        mock_response = MagicMock()
        mock_response.json.return_value = {"embedding": [0.1, 0.2, 0.3]}
        mock_response.raise_for_status.return_value = None
        
        with patch("httpx.AsyncClient.post", new=AsyncMock(return_value=mock_response)):
            vec = await embed_text("Hello")
        assert vec == [0.1, 0.2, 0.3]

    @pytest.mark.asyncio
    async def test_embed_text_http_error(self):
        mock_response = MagicMock()
        mock_response.status_code = 500
        error = httpx.HTTPStatusError("500", request=MagicMock(), response=mock_response)
        
        with patch("httpx.AsyncClient.post", new=AsyncMock(side_effect=error)):
            with pytest.raises(RuntimeError, match="Embedding failed"):
                await embed_text("Hello")
                
    @pytest.mark.asyncio
    async def test_embed_text_connection_error(self):
        with patch("httpx.AsyncClient.post", new=AsyncMock(side_effect=httpx.RequestError("offline"))), \
             patch("app.vector_store.asyncio.sleep", new=AsyncMock()):
            with pytest.raises(RuntimeError, match="unavailable"):
                await embed_text("Hello")

    @pytest.mark.asyncio
    async def test_embed_text_retries_on_connection_error(self):
        from app.vector_store import _EMBED_MAX_RETRIES
        mock_post = AsyncMock(side_effect=httpx.RequestError("transient"))
        with patch("httpx.AsyncClient.post", mock_post), \
             patch("app.vector_store.asyncio.sleep", new=AsyncMock()):
            with pytest.raises(RuntimeError, match="unavailable"):
                await embed_text("Hello")
        assert mock_post.call_count == _EMBED_MAX_RETRIES

    @pytest.mark.asyncio
    async def test_embed_text_succeeds_on_retry(self):
        """If the first call fails but the second succeeds, the vector is returned."""
        good_response = MagicMock()
        good_response.json.return_value = {"embedding": [0.9, 0.8]}
        good_response.raise_for_status.return_value = None
        mock_post = AsyncMock(side_effect=[httpx.RequestError("blip"), good_response])
        with patch("httpx.AsyncClient.post", mock_post), \
             patch("app.vector_store.asyncio.sleep", new=AsyncMock()):
            vec = await embed_text("Hello")
        assert vec == [0.9, 0.8]
        assert mock_post.call_count == 2

    @pytest.mark.asyncio
    async def test_embed_texts_batch(self):
        mock_response = MagicMock()
        mock_response.json.return_value = {"embedding": [0.5]}
        mock_response.raise_for_status.return_value = None
        
        with patch("httpx.AsyncClient.post", new=AsyncMock(return_value=mock_response)):
            vecs = await embed_texts(["A", "B"])
        assert len(vecs) == 2
        assert vecs[0] == [0.5]
        assert vecs[1] == [0.5]


class TestCollectionManagement:
    @pytest.mark.asyncio
    async def test_ensure_collection_exists_creates_if_missing(self):
        mock_client = AsyncMock()
        # Qdrant client get_collections().collections return
        mock_collections = MagicMock()
        mock_collections.collections = []
        mock_client.get_collections.return_value = mock_collections
        
        with patch("app.vector_store._qdrant", mock_client):
            await ensure_collection_exists()
            mock_client.create_collection.assert_called_once()

    @pytest.mark.asyncio
    async def test_ensure_collection_exists_skips_if_present(self):
        mock_client = AsyncMock()
        mock_collection = MagicMock()
        from app.config import get_settings
        mock_collection.name = get_settings().qdrant_collection
        mock_collections = MagicMock()
        mock_collections.collections = [mock_collection]
        mock_client.get_collections.return_value = mock_collections
        
        with patch("app.vector_store._qdrant", mock_client):
            await ensure_collection_exists()
            mock_client.create_collection.assert_not_called()

    @pytest.mark.asyncio
    async def test_collection_info_returns_stats(self):
        mock_client = AsyncMock()
        mock_info = MagicMock()
        mock_info.points_count = 42
        mock_client.get_collection.return_value = mock_info
        
        with patch("app.vector_store._qdrant", mock_client):
            info = await collection_info()
            assert info["qdrant_connected"] is True
            assert info["vector_count"] == 42
            
    @pytest.mark.asyncio
    async def test_collection_info_returns_false_on_error(self):
        mock_client = AsyncMock()
        mock_client.get_collection.side_effect = Exception("Down")
        
        with patch("app.vector_store._qdrant", mock_client):
            info = await collection_info()
            assert info["qdrant_connected"] is False
            assert info["vector_count"] == 0


class TestIngestion:
    @pytest.mark.asyncio
    async def test_ingest_gold_standard_bullets(self):
        bullets = [{"text": "First bullet", "role_type": "data_science"}]
        mock_client = AsyncMock()
        
        with patch("app.vector_store.ensure_collection_exists", new=AsyncMock()), \
             patch("app.vector_store.embed_texts", new=AsyncMock(return_value=[[0.1, 0.2]])), \
             patch("app.vector_store._qdrant", mock_client):
            
            count = await ingest_gold_standard_bullets(bullets)
            
        assert count == 1
        mock_client.upsert.assert_called_once()


class TestRetrieval:
    @pytest.mark.asyncio
    async def test_retrieve_style_exemplars(self):
        from qdrant_client.http.models import ScoredPoint
        mock_client = AsyncMock()
        mock_point = ScoredPoint(
            id="123",
            version=1,
            score=0.9,
            payload={
                "text": "Gold standard bullet",
                "role_type": "data_science",
                "uses_separator": "|",
            }
        )
        mock_response = MagicMock()
        mock_response.points = [mock_point]
        mock_client.query_points.return_value = mock_response

        with patch("app.vector_store.embed_text", new=AsyncMock(return_value=[0.1])), \
             patch("app.vector_store._qdrant", mock_client):

            exemplars = await retrieve_style_exemplars("query param")

        assert len(exemplars) == 1
        assert exemplars[0].similarity_score == 0.9
        assert exemplars[0].uses_separator == "|"
        assert exemplars[0].role_type == RoleType.DATA_SCIENCE

    @pytest.mark.asyncio
    async def test_invalid_role_type_in_payload_defaults_to_general(self):
        """N6: legacy/corrupted payloads with bad role_type must not crash retrieval.

        Pre-fix, RoleType('data_sciecne') raised ValueError and the entire
        list-comprehension died, breaking /optimize for everyone.
        """
        from qdrant_client.http.models import ScoredPoint
        mock_client = AsyncMock()
        mock_point = ScoredPoint(
            id="999",
            version=1,
            score=0.7,
            payload={"text": "Bullet", "role_type": "data_sciecne"},  # typo
        )
        mock_response = MagicMock()
        mock_response.points = [mock_point]
        mock_client.query_points.return_value = mock_response

        with patch("app.vector_store.embed_text", new=AsyncMock(return_value=[0.1])), \
             patch("app.vector_store._qdrant", mock_client):
            exemplars = await retrieve_style_exemplars("q")

        assert len(exemplars) == 1
        assert exemplars[0].role_type == RoleType.GENERAL  # graceful default
