"""Integration tests for the MCP server adapter.

Tests the MCP tools end-to-end — TF-IDF (default) and Ollama (real embeddings).
"""

from __future__ import annotations

import json

import pytest

try:
    from mcp.server.fastmcp import FastMCP

    _HAS_MCP = True
except ImportError:
    _HAS_MCP = False

pytestmark = pytest.mark.skipif(not _HAS_MCP, reason="mcp package not installed")

from zettelkasten_memory.adapters.mcp_server import create_mcp_server, _build_backend
from zettelkasten_memory.backends import EmbeddingBackend, TfidfBackend


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


@pytest.fixture
def mcp_server(tmp_path):
    """MCP server with default TF-IDF backend."""
    persist = str(tmp_path / "test_memory.json")
    return create_mcp_server(persist_path=persist, name="test-zettel")


@pytest.fixture
def mcp_server_ollama(tmp_path):
    """MCP server with real Ollama embeddings."""
    persist = str(tmp_path / "ollama_memory.json")
    backend = _build_backend(
        provider="ollama",
        model="nomic-embed-text",
        token=None,
        account=None,
        base_url=None,
        compression=False,
    )
    return create_mcp_server(persist_path=persist, name="test-zettel-ollama", backend=backend)


@pytest.fixture
def mcp_server_compressed(tmp_path):
    """MCP server with Ollama + TurboQuant compression."""
    persist = str(tmp_path / "compressed_memory.json")
    backend = _build_backend(
        provider="ollama",
        model="nomic-embed-text",
        token=None,
        account=None,
        base_url=None,
        compression=True,
    )
    return create_mcp_server(persist_path=persist, name="test-zettel-compressed", backend=backend)


async def call_tool(server: FastMCP, name: str, arguments: dict | None = None) -> str:
    """Call an MCP tool and return the text content."""
    result = await server.call_tool(name, arguments or {})
    content_list = result[0] if isinstance(result, tuple) else result
    return content_list[0].text


# ------------------------------------------------------------------
# _build_backend tests
# ------------------------------------------------------------------


def test_build_backend_default():
    backend = _build_backend(None, None, None, None, None, False)
    assert isinstance(backend, TfidfBackend)


def test_build_backend_tfidf():
    backend = _build_backend("tfidf", None, None, None, None, False)
    assert isinstance(backend, TfidfBackend)


def test_build_backend_ollama():
    backend = _build_backend("ollama", "nomic-embed-text", None, None, None, False)
    assert isinstance(backend, EmbeddingBackend)


def test_build_backend_ollama_compressed():
    backend = _build_backend("ollama", None, None, None, None, True)
    assert isinstance(backend, EmbeddingBackend)
    assert backend._compressor is not None


def test_build_backend_openai_with_token():
    backend = _build_backend("openai", None, "sk-test", None, None, False)
    assert isinstance(backend, EmbeddingBackend)
    assert backend._embed_fn._api_key == "sk-test"


def test_build_backend_snowflake_with_token():
    backend = _build_backend("snowflake", None, "pat-secret", "org-acct", None, False)
    assert isinstance(backend, EmbeddingBackend)
    assert backend._embed_fn._token == "pat-secret"
    assert backend._embed_fn._account == "org-acct"


# ------------------------------------------------------------------
# TF-IDF server tests (default backend)
# ------------------------------------------------------------------


@pytest.mark.asyncio
async def test_memory_store_and_get(mcp_server):
    store_result = await call_tool(mcp_server, "memory_store", {
        "content": "The project uses FastAPI and PostgreSQL",
        "tags": ["architecture", "backend"],
        "importance": 0.8,
    })
    data = json.loads(store_result)
    assert "id" in data
    assert "architecture" in data["tags"]

    get_result = await call_tool(mcp_server, "memory_get", {"memory_id": data["id"]})
    memory = json.loads(get_result)
    assert memory["content"] == "The project uses FastAPI and PostgreSQL"
    assert memory["importance"] == 0.8


@pytest.mark.asyncio
async def test_memory_search(mcp_server):
    await call_tool(mcp_server, "memory_store", {"content": "FastAPI is used for the REST API layer"})
    await call_tool(mcp_server, "memory_store", {"content": "PostgreSQL is the primary database"})
    await call_tool(mcp_server, "memory_store", {"content": "The user prefers dark mode in the IDE"})

    results = json.loads(await call_tool(mcp_server, "memory_search", {
        "query": "what database does the project use?", "limit": 3,
    }))
    assert len(results) > 0
    assert any("PostgreSQL" in r["content"] for r in results)


@pytest.mark.asyncio
async def test_memory_delete(mcp_server):
    r = await call_tool(mcp_server, "memory_store", {"content": "temporary note to delete"})
    memory_id = json.loads(r)["id"]

    assert json.loads(await call_tool(mcp_server, "memory_delete", {"memory_id": memory_id}))["deleted"]
    assert "error" in json.loads(await call_tool(mcp_server, "memory_get", {"memory_id": memory_id}))
    assert not json.loads(await call_tool(mcp_server, "memory_delete", {"memory_id": memory_id}))["deleted"]


@pytest.mark.asyncio
async def test_memory_connections(mcp_server):
    r1 = await call_tool(mcp_server, "memory_store", {"content": "FastAPI handles REST API requests"})
    r2 = await call_tool(mcp_server, "memory_store", {"content": "REST API endpoints use JWT auth"})

    connected = json.loads(await call_tool(mcp_server, "memory_connections", {
        "memory_id": json.loads(r1)["id"], "depth": 1,
    }))
    assert isinstance(connected, list)


@pytest.mark.asyncio
async def test_memory_stats(mcp_server):
    assert json.loads(await call_tool(mcp_server, "memory_stats"))["total_zettels"] == 0

    await call_tool(mcp_server, "memory_store", {"content": "first memory"})
    await call_tool(mcp_server, "memory_store", {"content": "second memory"})
    assert json.loads(await call_tool(mcp_server, "memory_stats"))["total_zettels"] == 2


@pytest.mark.asyncio
async def test_memory_store_with_metadata(mcp_server):
    r = await call_tool(mcp_server, "memory_store", {
        "content": "deployment uses Docker",
        "metadata": {"source": "team-handbook", "confidence": "high"},
    })
    memory = json.loads(await call_tool(mcp_server, "memory_get", {"memory_id": json.loads(r)["id"]}))
    assert memory["metadata"]["source"] == "team-handbook"


@pytest.mark.asyncio
async def test_persistence_across_servers(tmp_path):
    persist = str(tmp_path / "persist.json")

    server1 = create_mcp_server(persist_path=persist)
    await call_tool(server1, "memory_store", {"content": "persistent fact"})
    assert json.loads(await call_tool(server1, "memory_stats"))["total_zettels"] == 1

    server2 = create_mcp_server(persist_path=persist)
    assert json.loads(await call_tool(server2, "memory_stats"))["total_zettels"] == 1

    results = json.loads(await call_tool(server2, "memory_search", {"query": "persistent"}))
    assert "persistent fact" in results[0]["content"]


# ------------------------------------------------------------------
# Ollama embedding server tests (real semantic search)
# ------------------------------------------------------------------


@pytest.mark.asyncio
async def test_ollama_server_semantic_search(mcp_server_ollama):
    """MCP server with Ollama returns semantically correct results."""
    await call_tool(mcp_server_ollama, "memory_store", {
        "content": "The application is deployed using Docker containers on AWS ECS",
    })
    await call_tool(mcp_server_ollama, "memory_store", {
        "content": "Authentication uses JSON Web Tokens for session management",
    })
    await call_tool(mcp_server_ollama, "memory_store", {
        "content": "The frontend is built with React and TypeScript",
    })

    results = json.loads(await call_tool(mcp_server_ollama, "memory_search", {
        "query": "how do we ship the app to production?",
    }))
    assert len(results) > 0
    assert "Docker" in results[0]["content"]


@pytest.mark.asyncio
async def test_ollama_server_store_search_delete_cycle(mcp_server_ollama):
    """Full lifecycle with real embeddings."""
    r = await call_tool(mcp_server_ollama, "memory_store", {
        "content": "PostgreSQL is the primary relational database",
        "tags": ["database"],
        "importance": 0.9,
    })
    memory_id = json.loads(r)["id"]

    results = json.loads(await call_tool(mcp_server_ollama, "memory_search", {
        "query": "what database engine do we use?",
    }))
    assert any("PostgreSQL" in r["content"] for r in results)

    assert json.loads(await call_tool(mcp_server_ollama, "memory_delete", {"memory_id": memory_id}))["deleted"]
    assert json.loads(await call_tool(mcp_server_ollama, "memory_stats"))["total_zettels"] == 0


# ------------------------------------------------------------------
# Compressed server tests
# ------------------------------------------------------------------


@pytest.mark.asyncio
async def test_compressed_server_search(mcp_server_compressed):
    """MCP server with compression still returns correct results."""
    await call_tool(mcp_server_compressed, "memory_store", {
        "content": "Machine learning models are trained on GPU clusters",
    })
    await call_tool(mcp_server_compressed, "memory_store", {
        "content": "The CI/CD pipeline runs on GitHub Actions",
    })

    results = json.loads(await call_tool(mcp_server_compressed, "memory_search", {
        "query": "how do we train AI models?",
    }))
    assert len(results) > 0
    assert "GPU" in results[0]["content"]
