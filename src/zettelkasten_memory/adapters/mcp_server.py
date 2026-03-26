"""
MCP server adapter — expose ZettelMemory as tools for Claude Code and other MCP clients.

Run as a standalone server:
    python -m zettelkasten_memory.adapters.mcp_server --persist memory.json

Or configure in Claude Code's settings (~/.claude/settings.json):
    {
      "mcpServers": {
        "zettel-memory": {
          "command": "pixi",
          "args": ["run", "-e", "mcp", "python", "-m",
                   "zettelkasten_memory.adapters.mcp_server",
                   "--persist", "/path/to/memory.json"]
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
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any

from zettelkasten_memory.core import ZettelMemory

try:
    from mcp.server.fastmcp import FastMCP

    _HAS_MCP = True
except ImportError:
    _HAS_MCP = False


def create_mcp_server(
    persist_path: str | None = None,
    name: str = "zettel-memory",
) -> Any:
    """Create an MCP server exposing ZettelMemory tools."""
    if not _HAS_MCP:
        raise ImportError("mcp is required. Install with: pixi add mcp")

    mem = ZettelMemory()
    if persist_path:
        try:
            mem = ZettelMemory.load(persist_path)
        except (FileNotFoundError, Exception):
            pass

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
    parser = argparse.ArgumentParser(description="Zettelkasten Memory MCP Server")
    parser.add_argument("--persist", type=str, default=None, help="Path to JSON file for persistence")
    parser.add_argument("--name", type=str, default="zettel-memory", help="Server name")
    args = parser.parse_args()

    server = create_mcp_server(persist_path=args.persist, name=args.name)
    server.run()


if __name__ == "__main__":
    main()
