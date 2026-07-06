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

from zettelkasten_memory.adapters._tools import build_backend
from zettelkasten_memory.adapters.mcp_server import create_mcp_server
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
def mcp_server_ollama(tmp_path, ollama_embed_fn):
    """MCP server with real local embeddings."""
    persist = str(tmp_path / "ollama_memory.json")
    backend = EmbeddingBackend(embed_fn=ollama_embed_fn)
    return create_mcp_server(persist_path=persist, name="test-zettel-ollama", backend=backend)


@pytest.fixture
def mcp_server_compressed(tmp_path, ollama_embed_fn):
    """MCP server with real local embeddings + TurboQuant compression."""
    from zettelkasten_memory.compression import TurboQuantCompressor

    persist = str(tmp_path / "compressed_memory.json")
    backend = EmbeddingBackend(embed_fn=ollama_embed_fn, compressor=TurboQuantCompressor())
    return create_mcp_server(persist_path=persist, name="test-zettel-compressed", backend=backend)


async def call_tool(server: FastMCP, name: str, arguments: dict | None = None) -> str:
    """Call an MCP tool and return the text content."""
    result = await server.call_tool(name, arguments or {})
    content_list = result[0] if isinstance(result, tuple) else result
    return content_list[0].text


# ------------------------------------------------------------------
# build_backend tests
# ------------------------------------------------------------------


def test_build_backend_default():
    assert isinstance(build_backend(None), TfidfBackend)


def test_build_backend_tfidf():
    assert isinstance(build_backend("tfidf"), TfidfBackend)


def test_build_backend_ollama():
    backend = build_backend("ollama", model="nomic-embed-text")
    assert isinstance(backend, EmbeddingBackend)


def test_build_backend_ollama_compressed():
    backend = build_backend("ollama", compression=True)
    assert isinstance(backend, EmbeddingBackend)
    assert backend._compressor is not None


def test_build_backend_openai_env_token(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    backend = build_backend("openai")
    assert isinstance(backend, EmbeddingBackend)
    assert backend._embed_fn._api_key == "sk-test"


def test_build_backend_snowflake_env_token(monkeypatch):
    monkeypatch.setenv("SNOWFLAKE_PAT_TOKEN", "pat-secret")
    backend = build_backend("snowflake", account="org-acct")
    assert isinstance(backend, EmbeddingBackend)
    assert backend._embed_fn._token == "pat-secret"
    assert backend._embed_fn._account == "org-acct"


def test_build_backend_malgra():
    backend = build_backend("malgra", base_url="http://127.0.0.1:9999")
    assert isinstance(backend, EmbeddingBackend)
    assert backend._embed_fn.base_url == "http://127.0.0.1:9999"


# ------------------------------------------------------------------
# Hardening tests
# ------------------------------------------------------------------


def test_token_flag_removed():
    from zettelkasten_memory.adapters.mcp_server import main

    with pytest.raises(SystemExit):
        import sys
        from unittest.mock import patch

        with patch.object(sys, "argv", ["mcp_server", "--token", "sk-leaky"]):
            main()


def test_encrypt_flag_requires_key(monkeypatch):
    import sys
    from unittest.mock import patch

    from zettelkasten_memory.adapters.mcp_server import main

    for var in ("ZETTEL_MEMORY_KEY", "ZETTEL_MEMORY_KEY_FILE", "ZETTEL_MEMORY_PASSPHRASE"):
        monkeypatch.delenv(var, raising=False)
    with pytest.raises(SystemExit):
        with patch.object(sys, "argv", ["mcp_server", "--encrypt"]):
            main()


def test_corrupt_store_raises_instead_of_clobbering(tmp_path):
    persist = tmp_path / "corrupt.json"
    persist.write_text("{ not valid json !!!")
    with pytest.raises(RuntimeError, match="corrupt or unreadable"):
        create_mcp_server(persist_path=str(persist))
    # the corrupt file was NOT overwritten
    assert persist.read_text() == "{ not valid json !!!"


@pytest.mark.asyncio
async def test_namespace_binding(tmp_path):
    persist = str(tmp_path / "ns.json")
    server_a = create_mcp_server(persist_path=persist, namespace="tenant-a")
    await call_tool(server_a, "memory_store", {"content": "alpha secret plans"})

    server_b = create_mcp_server(persist_path=persist, namespace="tenant-b")
    results = json.loads(
        await call_tool(server_b, "memory_search", {"query": "alpha secret plans"})
    )
    assert results == []
    stats = json.loads(await call_tool(server_b, "memory_stats"))
    assert stats["namespace_zettels"] == 0


@pytest.mark.asyncio
async def test_encrypted_persistence(tmp_path, monkeypatch):
    pytest.importorskip("cryptography")
    import os as _os

    from zettelkasten_memory.crypto import MAGIC

    monkeypatch.setenv("ZETTEL_MEMORY_KEY", _os.urandom(32).hex())
    persist = tmp_path / "enc.bin"
    server = create_mcp_server(persist_path=str(persist), encrypt=True)
    await call_tool(server, "memory_store", {"content": "classified memo"})
    raw = persist.read_bytes()
    assert raw.startswith(MAGIC)
    assert b"classified memo" not in raw


@pytest.mark.asyncio
async def test_oversized_content_returns_error_json(tmp_path, monkeypatch):
    monkeypatch.setenv("ZETTEL_MAX_CONTENT_BYTES", "64")
    server = create_mcp_server(persist_path=str(tmp_path / "m.json"))
    result = json.loads(await call_tool(server, "memory_store", {"content": "x" * 200}))
    assert "error" in result and "limit" in result["error"]


# ------------------------------------------------------------------
# TF-IDF server tests (default backend)
# ------------------------------------------------------------------


@pytest.mark.asyncio
async def test_memory_store_and_get(mcp_server):
    store_result = await call_tool(
        mcp_server,
        "memory_store",
        {
            "content": "The project uses FastAPI and PostgreSQL",
            "tags": ["architecture", "backend"],
            "importance": 0.8,
        },
    )
    data = json.loads(store_result)
    assert "id" in data
    assert "architecture" in data["tags"]

    get_result = await call_tool(mcp_server, "memory_get", {"memory_id": data["id"]})
    memory = json.loads(get_result)
    assert memory["content"] == "The project uses FastAPI and PostgreSQL"
    assert memory["importance"] == 0.8


@pytest.mark.asyncio
async def test_memory_search(mcp_server):
    await call_tool(
        mcp_server, "memory_store", {"content": "FastAPI is used for the REST API layer"}
    )
    await call_tool(mcp_server, "memory_store", {"content": "PostgreSQL is the primary database"})
    await call_tool(
        mcp_server, "memory_store", {"content": "The user prefers dark mode in the IDE"}
    )

    results = json.loads(
        await call_tool(
            mcp_server,
            "memory_search",
            {
                "query": "what database does the project use?",
                "limit": 3,
            },
        )
    )
    assert len(results) > 0
    assert any("PostgreSQL" in r["content"] for r in results)


@pytest.mark.asyncio
async def test_memory_delete(mcp_server):
    r = await call_tool(mcp_server, "memory_store", {"content": "temporary note to delete"})
    memory_id = json.loads(r)["id"]

    assert json.loads(await call_tool(mcp_server, "memory_delete", {"memory_id": memory_id}))[
        "deleted"
    ]
    assert "error" in json.loads(
        await call_tool(mcp_server, "memory_get", {"memory_id": memory_id})
    )
    assert not json.loads(await call_tool(mcp_server, "memory_delete", {"memory_id": memory_id}))[
        "deleted"
    ]


@pytest.mark.asyncio
async def test_memory_connections(mcp_server):
    r1 = await call_tool(
        mcp_server, "memory_store", {"content": "FastAPI handles REST API requests"}
    )
    r2 = await call_tool(mcp_server, "memory_store", {"content": "REST API endpoints use JWT auth"})

    connected = json.loads(
        await call_tool(
            mcp_server,
            "memory_connections",
            {
                "memory_id": json.loads(r1)["id"],
                "depth": 1,
            },
        )
    )
    assert isinstance(connected, list)


@pytest.mark.asyncio
async def test_memory_stats(mcp_server):
    assert json.loads(await call_tool(mcp_server, "memory_stats"))["total_zettels"] == 0

    await call_tool(mcp_server, "memory_store", {"content": "first memory"})
    await call_tool(mcp_server, "memory_store", {"content": "second memory"})
    assert json.loads(await call_tool(mcp_server, "memory_stats"))["total_zettels"] == 2


@pytest.mark.asyncio
async def test_memory_store_with_metadata(mcp_server):
    r = await call_tool(
        mcp_server,
        "memory_store",
        {
            "content": "deployment uses Docker",
            "metadata": {"source": "team-handbook", "confidence": "high"},
        },
    )
    memory = json.loads(
        await call_tool(mcp_server, "memory_get", {"memory_id": json.loads(r)["id"]})
    )
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
    await call_tool(
        mcp_server_ollama,
        "memory_store",
        {
            "content": "The application is deployed using Docker containers on AWS ECS",
        },
    )
    await call_tool(
        mcp_server_ollama,
        "memory_store",
        {
            "content": "Authentication uses JSON Web Tokens for session management",
        },
    )
    await call_tool(
        mcp_server_ollama,
        "memory_store",
        {
            "content": "The frontend is built with React and TypeScript",
        },
    )

    results = json.loads(
        await call_tool(
            mcp_server_ollama,
            "memory_search",
            {
                "query": "how do we ship the app to production?",
            },
        )
    )
    assert len(results) > 0
    assert "Docker" in results[0]["content"]


@pytest.mark.asyncio
async def test_ollama_server_store_search_delete_cycle(mcp_server_ollama):
    """Full lifecycle with real embeddings."""
    r = await call_tool(
        mcp_server_ollama,
        "memory_store",
        {
            "content": "PostgreSQL is the primary relational database",
            "tags": ["database"],
            "importance": 0.9,
        },
    )
    memory_id = json.loads(r)["id"]

    results = json.loads(
        await call_tool(
            mcp_server_ollama,
            "memory_search",
            {
                "query": "what database engine do we use?",
            },
        )
    )
    assert any("PostgreSQL" in r["content"] for r in results)

    assert json.loads(
        await call_tool(mcp_server_ollama, "memory_delete", {"memory_id": memory_id})
    )["deleted"]
    assert json.loads(await call_tool(mcp_server_ollama, "memory_stats"))["total_zettels"] == 0


# ------------------------------------------------------------------
# Compressed server tests
# ------------------------------------------------------------------


@pytest.mark.asyncio
async def test_compressed_server_search(mcp_server_compressed):
    """MCP server with compression still returns correct results."""
    await call_tool(
        mcp_server_compressed,
        "memory_store",
        {
            "content": "Machine learning models are trained on GPU clusters",
        },
    )
    await call_tool(
        mcp_server_compressed,
        "memory_store",
        {
            "content": "The CI/CD pipeline runs on GitHub Actions",
        },
    )

    results = json.loads(
        await call_tool(
            mcp_server_compressed,
            "memory_search",
            {
                "query": "how do we train AI models?",
            },
        )
    )
    assert len(results) > 0
    assert "GPU" in results[0]["content"]
