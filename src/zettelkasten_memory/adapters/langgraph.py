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

from typing import Any, Iterable, Optional
from datetime import datetime, timezone

from zettelkasten_memory.core import ZettelMemory

try:
    from langgraph.store.base import BaseStore, Item, Result, SearchItem
    from langgraph.store.base import GetOp, PutOp, SearchOp, ListNamespacesOp

    _HAS_LANGGRAPH = True
except ImportError:
    _HAS_LANGGRAPH = False


def _ns_key(namespace: tuple[str, ...]) -> str:
    """Flatten a namespace tuple to a string prefix."""
    return "/".join(namespace)


if _HAS_LANGGRAPH:

    class ZettelStore(BaseStore):
        """
        LangGraph-compatible store backed by ZettelMemory.

        Namespaces are mapped to zettel metadata so memories from different
        users/threads stay isolated during search.
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
                except (FileNotFoundError, Exception):
                    pass

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
            target_id = f"{ns}:{key}"

            for z in self._mem._zettels.values():
                if z.metadata.get("_store_ns") == ns and z.metadata.get("_store_key") == key:
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
                if z.metadata.get("_store_ns") == ns and z.metadata.get("_store_key") == key
            ]
            for zid in to_delete:
                self._mem.delete(zid)

            if op.value is not None:
                content = op.value.get("content", str(op.value)) if isinstance(op.value, dict) else str(op.value)
                self._mem.add(
                    content,
                    metadata={
                        "_store_ns": ns,
                        "_store_key": key,
                        "_store_value": op.value,
                    },
                )
            self._persist()

        def _handle_search(self, op: SearchOp) -> list[SearchItem]:
            ns = _ns_key(op.namespace_prefix)
            query = op.query
            limit = op.limit or 10

            if query:
                results = self._mem.search(query, limit=limit * 2)
                # Filter to namespace
                filtered = [
                    r
                    for r in results
                    if r.zettel.metadata.get("_store_ns", "").startswith(ns)
                ][:limit]
            else:
                filtered_zettels = [
                    z
                    for z in self._mem._zettels.values()
                    if z.metadata.get("_store_ns", "").startswith(ns)
                ]
                from zettelkasten_memory.core import SearchResult

                filtered = [SearchResult(zettel=z, score=1.0) for z in filtered_zettels[:limit]]

            items: list[SearchItem] = []
            for r in filtered:
                z = r.zettel
                store_ns = z.metadata.get("_store_ns", "")
                items.append(
                    SearchItem(
                        value=z.metadata.get("_store_value", {"content": z.content}),
                        key=z.metadata.get("_store_key", z.id),
                        namespace=tuple(store_ns.split("/")) if store_ns else (),
                        created_at=datetime.fromtimestamp(z.created_at, tz=timezone.utc),
                        updated_at=datetime.fromtimestamp(z.accessed_at, tz=timezone.utc),
                        score=r.score,
                    )
                )
            return items

        def _handle_list_namespaces(self, op: ListNamespacesOp) -> list[tuple[str, ...]]:
            namespaces: set[tuple[str, ...]] = set()
            for z in self._mem._zettels.values():
                ns = z.metadata.get("_store_ns", "")
                if ns:
                    namespaces.add(tuple(ns.split("/")))
            return sorted(namespaces)

else:

    class ZettelStore:  # type: ignore[no-redef]
        """Placeholder — install langgraph to use this adapter."""

        def __init__(self, *args: Any, **kwargs: Any):
            raise ImportError(
                "langgraph is required for ZettelStore. "
                "Install it with: pixi add langgraph langgraph-checkpoint"
            )
