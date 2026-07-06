"""Shared memory-tool implementations for the MCP and SMCP server adapters.

Both servers expose the same six tools; the bodies live here so the two
protocol layers stay thin and identical in behavior.  Every function takes an
explicit keyword-only *namespace* — the server derives it from its own binding
(process-level for stdio MCP, authenticated identity for SMCP), never from a
client-supplied parameter.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any

from zettelkasten_memory.backends import EmbeddingBackend, TfidfBackend
from zettelkasten_memory.core import ZettelMemory

logger = logging.getLogger(__name__)

ENV_NAMESPACE = "ZETTEL_NAMESPACE"
ENV_MAX_CONTENT = "ZETTEL_MAX_CONTENT_BYTES"
ENV_MAX_METADATA = "ZETTEL_MAX_METADATA_BYTES"


def build_backend(
    provider: str | None,
    model: str | None = None,
    account: str | None = None,
    base_url: str | None = None,
    compression: bool = False,
):
    """Construct a search backend from server options.

    Credentials are read from environment variables by the providers
    themselves (OPENAI_API_KEY, SNOWFLAKE_PAT_TOKEN, MALGRA_AGENT_JWT, ...);
    secrets are never accepted as parameters here.
    """
    if provider is None or provider == "tfidf":
        return TfidfBackend()

    from zettelkasten_memory.providers import get_provider

    kwargs: dict[str, Any] = {}
    if model:
        kwargs["model"] = model
    if account and provider in ("snowflake", "cortex"):
        kwargs["account"] = account
    if base_url and provider in ("ollama", "malgra", "openai-compat"):
        kwargs["base_url"] = base_url

    embed_fn = get_provider(provider, **kwargs)

    compressor = None
    if compression:
        from zettelkasten_memory.compression import TurboQuantCompressor

        compressor = TurboQuantCompressor()

    return EmbeddingBackend(embed_fn=embed_fn, compressor=compressor)


def build_memory(
    persist_path: str | None,
    backend=None,
    *,
    camouflage=None,
) -> ZettelMemory:
    """Create (or load) the ZettelMemory instance a server operates on.

    A missing store file starts fresh; any other load failure (corrupt JSON,
    wrong encryption key, unreadable file) raises instead of silently booting
    an empty memory that would overwrite the store on the next persist.
    """
    size_kwargs = {}
    if os.environ.get(ENV_MAX_CONTENT):
        size_kwargs["max_content_bytes"] = int(os.environ[ENV_MAX_CONTENT])
    if os.environ.get(ENV_MAX_METADATA):
        size_kwargs["max_metadata_bytes"] = int(os.environ[ENV_MAX_METADATA])

    if persist_path:
        try:
            mem = ZettelMemory.load(persist_path, camouflage=camouflage)
            for attr, value in size_kwargs.items():
                setattr(mem, attr, value)
            if backend is not None and isinstance(backend, EmbeddingBackend):
                mem._backend = backend
                mem._backend.needs_rebuild = True
            return mem
        except FileNotFoundError:
            logger.info("no store at %s, starting fresh", persist_path)
        except (json.JSONDecodeError, KeyError, ValueError, OSError) as exc:
            raise RuntimeError(f"corrupt or unreadable memory store at {persist_path}") from exc

    return ZettelMemory(backend=backend, camouflage=camouflage, **size_kwargs)


def persist_memory(
    mem: ZettelMemory, persist_path: str | None, encrypt: bool | str = "auto"
) -> None:
    if persist_path:
        mem.save(persist_path, encrypt=encrypt)


# ------------------------------------------------------------------
# Tool bodies (shared by MCP and SMCP)
# ------------------------------------------------------------------


def store(
    mem: ZettelMemory,
    content: str,
    tags: list[str] | None = None,
    importance: float = 0.5,
    metadata: dict[str, str] | None = None,
    *,
    namespace: str,
) -> dict[str, Any]:
    zettel = mem.add(
        content,
        tags=set(tags) if tags else None,
        importance=importance,
        metadata=metadata or {},
        namespace=namespace,
    )
    return {
        "id": zettel.id,
        "connections": sorted(zettel.connections),
        "tags": sorted(zettel.tags),
    }


def search(
    mem: ZettelMemory, query: str, limit: int = 5, *, namespace: str
) -> list[dict[str, Any]]:
    results = mem.search(query, limit=limit, namespace=namespace)
    return [
        {
            "id": r.zettel.id,
            "content": r.zettel.content,
            "score": round(r.score, 4),
            "tags": sorted(r.zettel.tags),
            "connections": len(r.zettel.connections),
        }
        for r in results
    ]


def get(mem: ZettelMemory, memory_id: str, *, namespace: str) -> dict[str, Any]:
    z = mem.get(memory_id, namespace=namespace)
    if z is None:
        return {"error": "Memory not found"}
    return z.to_dict()


def delete(mem: ZettelMemory, memory_id: str, *, namespace: str) -> dict[str, Any]:
    return {"deleted": mem.delete(memory_id, namespace=namespace)}


def connections(
    mem: ZettelMemory, memory_id: str, depth: int = 1, *, namespace: str
) -> list[dict[str, Any]]:
    connected = mem.get_connected(memory_id, depth=depth, namespace=namespace)
    return [{"id": z.id, "content": z.content, "tags": sorted(z.tags)} for z in connected]


def stats(mem: ZettelMemory, *, namespace: str) -> dict[str, Any]:
    s = mem.stats
    s["namespace_zettels"] = s.get("namespaces", {}).get(namespace, 0)
    return s


# SMCP capability descriptions (JSON-schema-flavored, mirrors the MCP tools)
TOOL_SPECS: dict[str, dict[str, Any]] = {
    "memory_store": {
        "description": "Store a new memory. Returns the memory ID.",
        "parameters": {
            "content": {"type": "string", "required": True},
            "tags": {"type": "array", "items": "string"},
            "importance": {"type": "number", "default": 0.5},
            "metadata": {"type": "object"},
        },
    },
    "memory_search": {
        "description": "Search memories by semantic similarity.",
        "parameters": {
            "query": {"type": "string", "required": True},
            "limit": {"type": "integer", "default": 5},
        },
    },
    "memory_get": {
        "description": "Get a specific memory by its ID.",
        "parameters": {"memory_id": {"type": "string", "required": True}},
    },
    "memory_delete": {
        "description": "Delete a memory by ID. Also cleans up connections.",
        "parameters": {"memory_id": {"type": "string", "required": True}},
    },
    "memory_connections": {
        "description": "Get memories connected to the given one (graph traversal).",
        "parameters": {
            "memory_id": {"type": "string", "required": True},
            "depth": {"type": "integer", "default": 1},
        },
    },
    "memory_stats": {
        "description": "Get memory statistics — counts, connections, index state.",
        "parameters": {},
    },
}
