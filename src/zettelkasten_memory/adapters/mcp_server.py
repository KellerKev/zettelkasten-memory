"""
MCP server adapter — expose ZettelMemory as tools for Claude Code and other MCP clients.

Run as a standalone server:
    python -m zettelkasten_memory.adapters.mcp_server --persist memory.json

With embeddings (Ollama, local, free):
    python -m zettelkasten_memory.adapters.mcp_server --persist memory.json --provider ollama

Via an LLM gateway (zero secrets on this host):
    python -m zettelkasten_memory.adapters.mcp_server --persist memory.json \
        --provider malgra --base-url http://127.0.0.1:8766

Encrypted at rest + PII camouflage:
    export ZETTEL_MEMORY_KEY=<32-byte hex>
    export ZETTEL_PII_KEY=<64-byte hex>
    python -m zettelkasten_memory.adapters.mcp_server --persist memory.bin \
        --encrypt --camouflage

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
    - memory_reflect: Gather what memory knows about a topic (to summarize)
    - memory_prune: Find/delete stale, low-value memories (dry run by default)

Supported providers (--provider):
    tfidf               Zero config, no API keys (default)
    ollama              Local Ollama server (nomic-embed-text)
    malgra              OpenAI-compatible gateway / llama.cpp (dummy key)
    openai              Requires OPENAI_API_KEY
    cohere              Requires COHERE_API_KEY
    voyage              Requires VOYAGE_API_KEY
    snowflake / cortex  Requires SNOWFLAKE_ACCOUNT + SNOWFLAKE_PAT_TOKEN
    sentence-transformers / local  Local model, no API key

Secrets are environment-only (never CLI flags):
    OPENAI_API_KEY          OpenAI API key
    COHERE_API_KEY          Cohere API key
    VOYAGE_API_KEY          Voyage AI API key
    SNOWFLAKE_ACCOUNT       Snowflake account identifier (orgname-accountname)
    SNOWFLAKE_PAT_TOKEN     Snowflake Programmatic Access Token
    MALGRA_API_KEY / MALGRA_AGENT_JWT   Gateway credentials (default: dummy)
    ZETTEL_MEMORY_KEY       32-byte AES key (hex/base64) for encryption at rest
    ZETTEL_MEMORY_KEY_FILE  Path to a key file (alternative to ZETTEL_MEMORY_KEY)
    ZETTEL_MEMORY_PASSPHRASE  Passphrase (scrypt-derived key) alternative
    ZETTEL_PII_KEY          32/48/64-byte AES-SIV key for PII camouflage
    ZETTEL_NAMESPACE        Namespace this server is bound to (default: "default")
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any

from zettelkasten_memory.adapters import _tools
from zettelkasten_memory.core import ZettelMemory

try:
    from mcp.server.fastmcp import FastMCP

    _HAS_MCP = True
except ImportError:
    _HAS_MCP = False


def create_mcp_server(
    persist_path: str | None = None,
    name: str = "zettel-memory",
    backend=None,
    *,
    namespace: str | None = None,
    encrypt: bool | str = "auto",
    camouflage=None,
) -> Any:
    """Create an MCP server exposing ZettelMemory tools.

    The server is bound to a single *namespace* (process-level binding —
    stdio MCP has no per-connection identity).  Namespace is never a tool
    parameter.
    """
    if not _HAS_MCP:
        raise ImportError("mcp is required. Install with: pixi add mcp")

    bound_ns = namespace or os.environ.get(_tools.ENV_NAMESPACE) or "default"
    mem = _tools.build_memory(persist_path, backend, camouflage=camouflage)

    mcp = FastMCP(name)

    def _persist() -> None:
        _tools.persist_memory(mem, persist_path, encrypt=encrypt)

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
        try:
            result = _tools.store(mem, content, tags, importance, metadata, namespace=bound_ns)
        except ValueError as exc:
            return json.dumps({"error": str(exc)})
        _persist()
        return json.dumps(result)

    @mcp.tool()
    def memory_search(query: str, limit: int = 5) -> str:
        """Search memories by semantic similarity.

        Returns the most relevant memories for the given query, ranked by
        a combination of text similarity, importance, recency, and connections.

        Args:
            query: What to search for (natural language).
            limit: Max number of results (default 5).
        """
        return json.dumps(_tools.search(mem, query, limit, namespace=bound_ns), indent=2)

    @mcp.tool()
    def memory_get(memory_id: str) -> str:
        """Get a specific memory by its ID.

        Args:
            memory_id: The ID returned when the memory was stored.
        """
        return json.dumps(_tools.get(mem, memory_id, namespace=bound_ns), indent=2)

    @mcp.tool()
    def memory_delete(memory_id: str) -> str:
        """Delete a memory by ID. Also cleans up connections.

        Args:
            memory_id: The ID of the memory to delete.
        """
        result = _tools.delete(mem, memory_id, namespace=bound_ns)
        _persist()
        return json.dumps(result)

    @mcp.tool()
    def memory_connections(memory_id: str, depth: int = 1) -> str:
        """Get memories connected to the given one (graph traversal).

        This follows the Zettelkasten link structure — memories are
        automatically linked when they are semantically similar.

        Args:
            memory_id: Starting memory ID.
            depth: How many hops to traverse (default 1).
        """
        return json.dumps(_tools.connections(mem, memory_id, depth, namespace=bound_ns), indent=2)

    @mcp.tool()
    def memory_stats() -> str:
        """Get memory statistics — total count, connections, index state."""
        return json.dumps(_tools.stats(mem, namespace=bound_ns), indent=2)

    @mcp.tool()
    def memory_reflect(topic: str, limit: int = 10) -> str:
        """Gather what memory knows about a topic, for you to summarize.

        Returns the provenance-wrapped context plus the top matching memories
        (tags, connectivity, score). Read-only — nothing is modified.

        Args:
            topic: The subject to reflect on (natural language).
            limit: Max memories to gather (default 10).
        """
        return json.dumps(_tools.reflect(mem, topic, limit, namespace=bound_ns), indent=2)

    @mcp.tool()
    def memory_prune(
        max_age_days: float | None = None,
        min_importance: float | None = None,
        limit: int = 20,
        dry_run: bool = True,
    ) -> str:
        """Find (and optionally delete) stale, low-value memories.

        A dry run by default: it reports candidates without deleting. Set
        dry_run=False to actually delete them (and clean up their links).

        Args:
            max_age_days: Only consider memories not accessed in this many days.
            min_importance: Only consider memories with importance below this.
            limit: Max candidates to return/delete (default 20).
            dry_run: When True (default), report only; when False, delete.
        """
        result = _tools.prune(mem, max_age_days, min_importance, limit, dry_run, namespace=bound_ns)
        if result.get("removed"):
            _persist()
        return json.dumps(result, indent=2)

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
  --provider malgra                        OpenAI-compatible gateway (dummy key)
  --provider malgra --base-url http://127.0.0.1:8092    llama.cpp etc.
  --provider openai                        Uses OPENAI_API_KEY env var
  --provider snowflake --account org-acct  Uses SNOWFLAKE_PAT_TOKEN env var
  --provider cohere                        Uses COHERE_API_KEY env var
  --provider voyage                        Uses VOYAGE_API_KEY env var

Security:
  --encrypt      requires ZETTEL_MEMORY_KEY / _KEY_FILE / _PASSPHRASE
  --camouflage   requires ZETTEL_PII_KEY
  --namespace    or ZETTEL_NAMESPACE env var

Secrets are read from environment variables only:
  OPENAI_API_KEY, COHERE_API_KEY, VOYAGE_API_KEY,
  SNOWFLAKE_ACCOUNT, SNOWFLAKE_PAT_TOKEN, MALGRA_API_KEY, MALGRA_AGENT_JWT,
  ZETTEL_MEMORY_KEY, ZETTEL_MEMORY_KEY_FILE, ZETTEL_MEMORY_PASSPHRASE,
  ZETTEL_PII_KEY
""",
    )
    parser.add_argument(
        "--persist",
        type=str,
        default=None,
        help="Path to store file for persistence",
    )
    parser.add_argument(
        "--name",
        type=str,
        default="zettel-memory",
        help="Server name (default: zettel-memory)",
    )
    parser.add_argument(
        "--provider",
        type=str,
        default=None,
        help="Embedding provider: tfidf, ollama, malgra, openai, cohere, voyage, snowflake, sentence-transformers",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model name for the chosen provider",
    )
    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--account",
        type=str,
        default=None,
        help="Snowflake account identifier (orgname-accountname)",
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default=None,
        help="Base URL for ollama/malgra providers",
    )
    parser.add_argument(
        "--compress",
        action="store_true",
        default=False,
        help="Enable TurboQuant vector compression (reduces storage 3-8x)",
    )
    parser.add_argument(
        "--namespace",
        type=str,
        default=None,
        help="Namespace this server is bound to (or ZETTEL_NAMESPACE; default: 'default')",
    )
    parser.add_argument(
        "--encrypt",
        action="store_true",
        default=False,
        help="Require AES-256-GCM encryption at rest (key from ZETTEL_MEMORY_* env vars)",
    )
    parser.add_argument(
        "--camouflage",
        action="store_true",
        default=False,
        help="Tokenize PII before indexing/persisting (key from ZETTEL_PII_KEY env var)",
    )
    parser.add_argument(
        "--no-detokenize",
        action="store_true",
        default=False,
        help="With --camouflage: return tokens instead of plaintext PII from tools",
    )

    args = parser.parse_args()

    if args.token:
        parser.error(
            "--token was removed: pass secrets via environment variables instead "
            "(OPENAI_API_KEY / COHERE_API_KEY / VOYAGE_API_KEY / SNOWFLAKE_PAT_TOKEN). "
            "CLI arguments leak via process listings and shell history."
        )

    encrypt: bool | str = "auto"
    if args.encrypt:
        from zettelkasten_memory import crypto

        if not crypto.encryption_available():
            parser.error(
                "--encrypt requires key material: set ZETTEL_MEMORY_KEY, "
                "ZETTEL_MEMORY_KEY_FILE, or ZETTEL_MEMORY_PASSPHRASE"
            )
        encrypt = True

    camouflage = None
    if args.camouflage:
        from zettelkasten_memory.camouflage import CamouflageCodec, CamouflageError

        try:
            camouflage = CamouflageCodec(reveal=not args.no_detokenize)
        except CamouflageError as exc:
            parser.error(str(exc))

    backend = _tools.build_backend(
        provider=args.provider,
        model=args.model,
        account=args.account,
        base_url=args.base_url,
        compression=args.compress,
    )

    server = create_mcp_server(
        persist_path=args.persist,
        name=args.name,
        backend=backend,
        namespace=args.namespace,
        encrypt=encrypt,
        camouflage=camouflage,
    )
    server.run()


if __name__ == "__main__":
    main()
