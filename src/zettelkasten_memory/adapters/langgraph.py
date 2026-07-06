"""
LangGraph adapter — plug ZettelMemory into LangGraph's Store API.

LangGraph uses BaseStore for long-term cross-thread memory. This module
provides ZettelStore which implements that interface, giving your LangGraph
agents Zettelkasten-style semantic memory with automatic linking.

Usage:
    from langgraph.graph import StateGraph
    from zettelkasten_memory.adapters.langgraph import ZettelStore

    store = ZettelStore(persist_path="langgraph_memory.json")

    graph = StateGraph(...)
    # ... define nodes/edges ...
    app = graph.compile(store=store)

    # In your node functions, use the store:
    async def my_node(state, *, store):
        # Search memory
        results = await store.asearch(("memories", "user_123"), query="project architecture")
        # Store memory
        await store.aput(("memories", "user_123"), "key1", {"content": "Uses FastAPI"})
"""

from __future__ import annotations

import json
import logging
from typing import Any, Iterable, Optional
from datetime import datetime, timezone

from zettelkasten_memory.core import ZettelMemory

logger = logging.getLogger(__name__)

try:
    from langgraph.store.base import BaseStore, Item, Result, SearchItem
    from langgraph.store.base import GetOp, PutOp, SearchOp, ListNamespacesOp

    _HAS_LANGGRAPH = True
except ImportError:
    _HAS_LANGGRAPH = False


def _ns_key(namespace: tuple[str, ...]) -> str:
    """Flatten a namespace tuple to a string prefix."""
    return "/".join(namespace)


def _ns_matches(zettel_ns: str, prefix: str) -> bool:
    """Exact or path-segment prefix match: 'a/b' matches 'a' but not 'a_x'."""
    if not prefix:
        return True
    return zettel_ns == prefix or zettel_ns.startswith(prefix + "/")


if _HAS_LANGGRAPH:

    class ZettelStore(BaseStore):
        """
        LangGraph-compatible store backed by ZettelMemory.

        Namespaces map to the first-class ``Zettel.namespace`` field, so
        isolation is enforced at the storage layer: search never returns and
        auto-linking never connects memories across namespaces.
        """

        def __init__(
            self,
            persist_path: str | None = None,
            max_zettels: int = 5000,
        ):
            self._persist_path = persist_path
            self._mem = ZettelMemory(max_zettels=max_zettels)

            if persist_path:
                try:
                    self._mem = ZettelMemory.load(persist_path)
                except FileNotFoundError:
                    logger.info("no store at %s, starting fresh", persist_path)
                except (json.JSONDecodeError, KeyError, ValueError, OSError) as exc:
                    raise RuntimeError(
                        f"corrupt or unreadable memory store at {persist_path}"
                    ) from exc
                self._migrate_legacy_namespaces()

        def _migrate_legacy_namespaces(self) -> None:
            """Move pre-namespace ``_store_ns`` metadata into Zettel.namespace."""
            migrated = 0
            for z in self._mem._zettels.values():
                legacy_ns = z.metadata.get("_store_ns")
                if legacy_ns and z.namespace == "default":
                    z.namespace = z.metadata.pop("_store_ns")
                    migrated += 1
            if migrated:
                logger.info("migrated %d zettels from metadata namespaces", migrated)

        def _persist(self) -> None:
            if self._persist_path:
                self._mem.save(self._persist_path)

        def batch(self, ops: Iterable) -> list:
            """Execute a batch of store operations synchronously."""
            results: list = []
            for op in ops:
                if isinstance(op, GetOp):
                    results.append(self._handle_get(op))
                elif isinstance(op, PutOp):
                    self._handle_put(op)
                    results.append(None)
                elif isinstance(op, SearchOp):
                    results.append(self._handle_search(op))
                elif isinstance(op, ListNamespacesOp):
                    results.append(self._handle_list_namespaces(op))
                else:
                    results.append(None)
            return results

        async def abatch(self, ops: Iterable) -> list:
            """Async version — delegates to sync since ZettelMemory is CPU-bound."""
            return self.batch(ops)

        def _handle_get(self, op: GetOp) -> Item | None:
            ns = _ns_key(op.namespace)
            key = op.key

            for z in self._mem._zettels.values():
                if z.namespace == ns and z.metadata.get("_store_key") == key:
                    return Item(
                        value=z.metadata.get("_store_value", {"content": z.content}),
                        key=key,
                        namespace=op.namespace,
                        created_at=datetime.fromtimestamp(z.created_at, tz=timezone.utc),
                        updated_at=datetime.fromtimestamp(z.accessed_at, tz=timezone.utc),
                    )
            return None

        def _handle_put(self, op: PutOp) -> None:
            ns = _ns_key(op.namespace)
            key = op.key

            # Delete existing entry with same ns:key
            to_delete = [
                zid
                for zid, z in self._mem._zettels.items()
                if z.namespace == ns and z.metadata.get("_store_key") == key
            ]
            for zid in to_delete:
                self._mem.delete(zid)

            if op.value is not None:
                content = (
                    op.value.get("content", str(op.value))
                    if isinstance(op.value, dict)
                    else str(op.value)
                )
                self._mem.add(
                    content,
                    metadata={
                        "_store_key": key,
                        "_store_value": op.value,
                    },
                    namespace=ns,
                )
            self._persist()

        def _handle_search(self, op: SearchOp) -> list[SearchItem]:
            ns = _ns_key(op.namespace_prefix)
            query = op.query
            limit = op.limit or 10

            if query:
                # namespace=None: prefix semantics need post-filtering across
                # namespaces; exact scoping is applied below with _ns_matches.
                results = self._mem.search(query, limit=limit * 2, namespace=None)
                filtered = [r for r in results if _ns_matches(r.zettel.namespace, ns)][:limit]
            else:
                filtered_zettels = [
                    z for z in self._mem._zettels.values() if _ns_matches(z.namespace, ns)
                ]
                from zettelkasten_memory.core import SearchResult

                filtered = [SearchResult(zettel=z, score=1.0) for z in filtered_zettels[:limit]]

            items: list[SearchItem] = []
            for r in filtered:
                z = r.zettel
                items.append(
                    SearchItem(
                        value=z.metadata.get("_store_value", {"content": z.content}),
                        key=z.metadata.get("_store_key", z.id),
                        namespace=tuple(z.namespace.split("/")) if z.namespace else (),
                        created_at=datetime.fromtimestamp(z.created_at, tz=timezone.utc),
                        updated_at=datetime.fromtimestamp(z.accessed_at, tz=timezone.utc),
                        score=r.score,
                    )
                )
            return items

        def _handle_list_namespaces(self, op: ListNamespacesOp) -> list[tuple[str, ...]]:
            namespaces: set[tuple[str, ...]] = set()
            for z in self._mem._zettels.values():
                if z.namespace and z.namespace != "default":
                    namespaces.add(tuple(z.namespace.split("/")))
            return sorted(namespaces)

else:

    class ZettelStore:  # type: ignore[no-redef]
        """Placeholder — install langgraph to use this adapter."""

        def __init__(self, *args: Any, **kwargs: Any):
            raise ImportError(
                "langgraph is required for ZettelStore. "
                "Install it with: pixi add langgraph langgraph-checkpoint"
            )
