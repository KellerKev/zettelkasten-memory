"""
MCP server adapter — expose ZettelMemory as tools for Claude Code and other MCP clients.

Run as a standalone server:
    python -m zettelkasten_memory.adapters.mcp_server --persist memory.json

With embeddings (Ollama, local, free):
    python -m zettelkasten_memory.adapters.mcp_server --persist memory.json --provider ollama

With Snowflake Cortex:
    python -m zettelkasten_memory.adapters.mcp_server --persist memory.json \
        --provider snowflake --model snowflake-arctic-embed-m-v1.5

Or configure in Claude Code's settings (~/.claude/settings.json):
    {
      "mcpServers": {
        "zettel-memory": {
          "command": "pixi",
          "args": ["run", "-e", "mcp", "python", "-m",
                   "zettelkasten_memory.adapters.mcp_server",
                   "--persist", "/path/to/memory.json",
                   "--provider", "ollama"],
          "cwd": "/path/to/zettelkasten-memory"
        }
      }
    }

Exposes these tools to the LLM:
    - memory_store: Save a new memory
    - memory_search: Search memories by query
    - memory_get: Get a specific memory by ID
    - memory_delete: Delete a memory
    - memory_connections: Get connected memories (graph traversal)
    - memory_stats: Get memory statistics

Supported providers (--provider):
    tfidf               Zero config, no API keys (default)
    ollama              Local Ollama server (nomic-embed-text)
    openai              Requires OPENAI_API_KEY
    cohere              Requires COHERE_API_KEY
    voyage              Requires VOYAGE_API_KEY
    snowflake / cortex  Requires SNOWFLAKE_ACCOUNT + SNOWFLAKE_PAT_TOKEN
    sentence-transformers / local  Local model, no API key

Environment variables for tokens:
    OPENAI_API_KEY          OpenAI API key
    COHERE_API_KEY          Cohere API key
    VOYAGE_API_KEY          Voyage AI API key
    SNOWFLAKE_ACCOUNT       Snowflake account identifier (orgname-accountname)
    SNOWFLAKE_PAT_TOKEN     Snowflake Programmatic Access Token
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any

from zettelkasten_memory.backends import EmbeddingBackend, TfidfBackend
from zettelkasten_memory.core import ZettelMemory

try:
    from mcp.server.fastmcp import FastMCP

    _HAS_MCP = True
except ImportError:
    _HAS_MCP = False


def _build_backend(
    provider: str | None,
    model: str | None,
    token: str | None,
    account: str | None,
    base_url: str | None,
    compression: bool,
):
    """Construct the appropriate backend from CLI flags."""
    if provider is None or provider == "tfidf":
        return TfidfBackend()

    from zettelkasten_memory.providers import get_provider

    # Build kwargs for the provider
    kwargs: dict[str, Any] = {}
    if model:
        kwargs["model"] = model

    # Token handling — CLI flag overrides env vars
    if token:
        if provider in ("openai",):
            kwargs["api_key"] = token
        elif provider in ("cohere",):
            kwargs["api_key"] = token
        elif provider in ("voyage",):
            kwargs["api_key"] = token
        elif provider in ("snowflake", "cortex"):
            kwargs["token"] = token

    # Snowflake account
    if account:
        kwargs["account"] = account

    # Ollama base URL
    if base_url and provider == "ollama":
        kwargs["base_url"] = base_url

    embed_fn = get_provider(provider, **kwargs)

    # Optional compression
    compressor = None
    if compression:
        from zettelkasten_memory.compression import TurboQuantCompressor

        compressor = TurboQuantCompressor()

    return EmbeddingBackend(embed_fn=embed_fn, compressor=compressor)


def create_mcp_server(
    persist_path: str | None = None,
    name: str = "zettel-memory",
    backend=None,
) -> Any:
    """Create an MCP server exposing ZettelMemory tools."""
    if not _HAS_MCP:
        raise ImportError("mcp is required. Install with: pixi add mcp")

    mem = ZettelMemory(backend=backend) if backend else ZettelMemory()
    if persist_path:
        try:
            mem = ZettelMemory.load(persist_path)
            # If we have a new backend with embeddings, swap it in
            if backend is not None and isinstance(backend, EmbeddingBackend):
                mem._backend = backend
                mem._backend.needs_rebuild = True
        except (FileNotFoundError, Exception):
            if backend:
                mem = ZettelMemory(backend=backend)

    mcp = FastMCP(name)

    def _persist() -> None:
        if persist_path:
            mem.save(persist_path)

    @mcp.tool()
    def memory_store(
        content: str,
        tags: list[str] | None = None,
        importance: float = 0.5,
        metadata: dict[str, str] | None = None,
    ) -> str:
        """Store a new memory. Returns the memory ID.

        Use this to remember facts, decisions, user preferences, project context,
        or anything that should persist across conversations.

        Args:
            content: The text to remember.
            tags: Optional labels for categorization (e.g. ["preference", "architecture"]).
            importance: 0.0 to 1.0 — how important this memory is (default 0.5).
            metadata: Optional key-value pairs for extra context.
        """
        zettel = mem.add(
            content,
            tags=set(tags) if tags else None,
            importance=importance,
            metadata=metadata or {},
        )
        _persist()
        return json.dumps(
            {"id": zettel.id, "connections": sorted(zettel.connections), "tags": sorted(zettel.tags)}
        )

    @mcp.tool()
    def memory_search(query: str, limit: int = 5) -> str:
        """Search memories by semantic similarity.

        Returns the most relevant memories for the given query, ranked by
        a combination of text similarity, importance, recency, and connections.

        Args:
            query: What to search for (natural language).
            limit: Max number of results (default 5).
        """
        results = mem.search(query, limit=limit)
        return json.dumps(
            [
                {
                    "id": r.zettel.id,
                    "content": r.zettel.content,
                    "score": round(r.score, 4),
                    "tags": sorted(r.zettel.tags),
                    "connections": len(r.zettel.connections),
                }
                for r in results
            ],
            indent=2,
        )

    @mcp.tool()
    def memory_get(memory_id: str) -> str:
        """Get a specific memory by its ID.

        Args:
            memory_id: The ID returned when the memory was stored.
        """
        z = mem.get(memory_id)
        if z is None:
            return json.dumps({"error": "Memory not found"})
        return json.dumps(z.to_dict(), indent=2)

    @mcp.tool()
    def memory_delete(memory_id: str) -> str:
        """Delete a memory by ID. Also cleans up connections.

        Args:
            memory_id: The ID of the memory to delete.
        """
        ok = mem.delete(memory_id)
        _persist()
        return json.dumps({"deleted": ok})

    @mcp.tool()
    def memory_connections(memory_id: str, depth: int = 1) -> str:
        """Get memories connected to the given one (graph traversal).

        This follows the Zettelkasten link structure — memories are
        automatically linked when they are semantically similar.

        Args:
            memory_id: Starting memory ID.
            depth: How many hops to traverse (default 1).
        """
        connected = mem.get_connected(memory_id, depth=depth)
        return json.dumps(
            [
                {
                    "id": z.id,
                    "content": z.content,
                    "tags": sorted(z.tags),
                }
                for z in connected
            ],
            indent=2,
        )

    @mcp.tool()
    def memory_stats() -> str:
        """Get memory statistics — total count, connections, index state."""
        return json.dumps(mem.stats, indent=2)

    return mcp


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Zettelkasten Memory MCP Server",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Provider examples:
  --provider tfidf                         TF-IDF (default, no API key)
  --provider ollama                        Local Ollama (nomic-embed-text)
  --provider ollama --model mxbai-embed-large
  --provider openai                        Uses OPENAI_API_KEY env var
  --provider openai --token sk-...         Explicit API key
  --provider snowflake --account org-acct  Uses SNOWFLAKE_PAT_TOKEN env var
  --provider snowflake --token pat-...     Explicit PAT token
  --provider cohere                        Uses COHERE_API_KEY env var
  --provider voyage                        Uses VOYAGE_API_KEY env var

Token env vars:
  OPENAI_API_KEY, COHERE_API_KEY, VOYAGE_API_KEY,
  SNOWFLAKE_ACCOUNT, SNOWFLAKE_PAT_TOKEN
""",
    )
    parser.add_argument(
        "--persist", type=str, default=None,
        help="Path to JSON file for persistence",
    )
    parser.add_argument(
        "--name", type=str, default="zettel-memory",
        help="Server name (default: zettel-memory)",
    )
    parser.add_argument(
        "--provider", type=str, default=None,
        help="Embedding provider: tfidf, ollama, openai, cohere, voyage, snowflake, sentence-transformers",
    )
    parser.add_argument(
        "--model", type=str, default=None,
        help="Model name for the chosen provider",
    )
    parser.add_argument(
        "--token", type=str, default=None,
        help="API key or access token (overrides env var for the chosen provider)",
    )
    parser.add_argument(
        "--account", type=str, default=None,
        help="Snowflake account identifier (orgname-accountname)",
    )
    parser.add_argument(
        "--base-url", type=str, default=None,
        help="Base URL for Ollama (default: http://localhost:11434)",
    )
    parser.add_argument(
        "--compress", action="store_true", default=False,
        help="Enable TurboQuant vector compression (reduces storage 3-8x)",
    )

    args = parser.parse_args()

    backend = _build_backend(
        provider=args.provider,
        model=args.model,
        token=args.token,
        account=args.account,
        base_url=args.base_url,
        compression=args.compress,
    )

    server = create_mcp_server(
        persist_path=args.persist,
        name=args.name,
        backend=backend,
    )
    server.run()


if __name__ == "__main__":
    main()
