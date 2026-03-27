"""
Core Zettelkasten memory engine.

Each memory is a "zettel" (note) with:
- Content and metadata
- Automatic tags extracted via the search backend
- Bidirectional links to related zettels (found by semantic similarity)
- Importance scoring based on access patterns and recency

Retrieval uses a pluggable search backend (TF-IDF by default, or embeddings)
with a keyword fallback.  Connected zettels are boosted in search results
(graph-aware ranking).
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from .backends import (
    EmbedFn,
    EmbeddingBackend,
    SearchBackend,
    TfidfBackend,
    backend_from_dict,
)


@dataclass
class Zettel:
    """A single note in the Zettelkasten."""

    id: str
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)
    tags: set[str] = field(default_factory=set)
    connections: set[str] = field(default_factory=set)
    created_at: float = field(default_factory=time.time)
    accessed_at: float = field(default_factory=time.time)
    access_count: int = 0
    importance: float = 0.5

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "content": self.content,
            "metadata": self.metadata,
            "tags": sorted(self.tags),
            "connections": sorted(self.connections),
            "created_at": self.created_at,
            "accessed_at": self.accessed_at,
            "access_count": self.access_count,
            "importance": self.importance,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Zettel:
        return cls(
            id=data["id"],
            content=data["content"],
            metadata=data.get("metadata", {}),
            tags=set(data.get("tags", [])),
            connections=set(data.get("connections", [])),
            created_at=data.get("created_at", time.time()),
            accessed_at=data.get("accessed_at", time.time()),
            access_count=data.get("access_count", 0),
            importance=data.get("importance", 0.5),
        )


@dataclass
class SearchResult:
    """A zettel with its relevance score."""

    zettel: Zettel
    score: float


class ZettelMemory:
    """
    Zettelkasten-inspired memory with semantic search and automatic linking.

    Usage:
        mem = ZettelMemory()
        mem.add("The user prefers concise answers", tags={"preference"})
        mem.add("Project uses FastAPI + PostgreSQL", tags={"architecture"})

        results = mem.search("what framework does the project use?")
        for r in results:
            print(r.score, r.zettel.content)

        # Persistence
        mem.save("memory.json")
        mem = ZettelMemory.load("memory.json")

        # Use embeddings instead of TF-IDF:
        from zettelkasten_memory.backends import EmbeddingBackend
        mem = ZettelMemory(backend=EmbeddingBackend(embed_fn=my_embed_fn))
    """

    def __init__(
        self,
        max_zettels: int = 5000,
        connection_threshold: float = 0.25,
        backend: SearchBackend | None = None,
    ):
        self.max_zettels = max_zettels
        self.connection_threshold = connection_threshold
        self._backend: SearchBackend = backend or TfidfBackend()
        self._zettels: dict[str, Zettel] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add(
        self,
        content: str,
        *,
        metadata: dict[str, Any] | None = None,
        tags: set[str] | None = None,
        importance: float = 0.5,
    ) -> Zettel:
        """Store a new zettel, auto-link to related zettels, and return it."""
        zid = self._make_id(content)
        now = time.time()

        zettel = Zettel(
            id=zid,
            content=content,
            metadata=metadata or {},
            tags=tags or set(),
            created_at=now,
            accessed_at=now,
            importance=max(0.0, min(1.0, importance)),
        )

        # Auto-extract additional tags from content
        zettel.tags.update(self._backend.extract_tags(content))

        # Find and create bidirectional links
        self._rebuild_index_if_needed()
        self._link_zettel(zettel)

        self._zettels[zid] = zettel
        self._backend.needs_rebuild = True

        # Evict if over capacity
        if len(self._zettels) > self.max_zettels:
            self._evict()

        return zettel

    def search(self, query: str, *, limit: int = 10, min_score: float = 0.0) -> list[SearchResult]:
        """Search for relevant zettels using the backend with graph boosting."""
        if not self._zettels:
            return []

        self._rebuild_index_if_needed()

        raw_pairs = self._backend.query(query)

        if not raw_pairs:
            return self._keyword_search(query, limit)

        # Score = similarity * importance * recency * connection_boost
        now = time.time()
        results: list[SearchResult] = []

        for zid, sim in raw_pairs:
            zettel = self._zettels.get(zid)
            if zettel is None:
                continue

            recency = 1.0 / (1.0 + (now - zettel.accessed_at) / 86400)  # decay over days
            conn_boost = 1.0 + 0.1 * min(len(zettel.connections), 10)  # cap boost at 2x
            score = sim * zettel.importance * (0.7 + 0.3 * recency) * conn_boost

            if score >= min_score:
                results.append(SearchResult(zettel=zettel, score=score))

        results.sort(key=lambda r: r.score, reverse=True)

        if not results:
            return self._keyword_search(query, limit)

        # Mark accessed
        for r in results[:limit]:
            r.zettel.access_count += 1
            r.zettel.accessed_at = now

        return results[:limit]

    def get(self, zettel_id: str) -> Zettel | None:
        """Get a zettel by ID."""
        return self._zettels.get(zettel_id)

    def delete(self, zettel_id: str) -> bool:
        """Delete a zettel and clean up its connections."""
        if zettel_id not in self._zettels:
            return False

        # Remove from other zettels' connections
        zettel = self._zettels[zettel_id]
        for cid in zettel.connections:
            if cid in self._zettels:
                self._zettels[cid].connections.discard(zettel_id)

        del self._zettels[zettel_id]
        self._backend.needs_rebuild = True
        return True

    def get_connected(self, zettel_id: str, *, depth: int = 1) -> list[Zettel]:
        """Get zettels connected to the given one, up to N hops."""
        if zettel_id not in self._zettels:
            return []

        visited: set[str] = {zettel_id}
        frontier = {zettel_id}

        for _ in range(depth):
            next_frontier: set[str] = set()
            for zid in frontier:
                z = self._zettels.get(zid)
                if z:
                    for cid in z.connections:
                        if cid not in visited and cid in self._zettels:
                            visited.add(cid)
                            next_frontier.add(cid)
            frontier = next_frontier
            if not frontier:
                break

        visited.discard(zettel_id)
        return [self._zettels[zid] for zid in visited]

    def get_context(self, query: str, *, max_tokens: int = 4000, limit: int = 10) -> str:
        """Search and format results as a context string for an LLM prompt."""
        results = self.search(query, limit=limit)
        if not results:
            return ""

        parts: list[str] = []
        est_tokens = 0

        for r in results:
            text = r.zettel.content
            tokens_est = len(text) // 3  # ~3 chars per token
            if est_tokens + tokens_est > max_tokens:
                break
            parts.append(text)
            est_tokens += tokens_est

        return "\n---\n".join(parts)

    @property
    def stats(self) -> dict[str, Any]:
        """Memory statistics."""
        total_connections = sum(len(z.connections) for z in self._zettels.values())
        return {
            "total_zettels": len(self._zettels),
            "total_connections": total_connections // 2,  # bidirectional, so halve
            "avg_connections": (total_connections / len(self._zettels)) if self._zettels else 0,
            "index_dirty": self._backend.needs_rebuild,
        }

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        """Save all zettels to a JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "version": 1,
            "zettels": [z.to_dict() for z in self._zettels.values()],
            "config": {
                "max_zettels": self.max_zettels,
                "connection_threshold": self.connection_threshold,
            },
            "backend": self._backend.to_dict(),
        }
        path.write_text(json.dumps(data, indent=2))

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        embed_fn: EmbedFn | None = None,
    ) -> ZettelMemory:
        """Load from a JSON file.

        If the saved memory used an ``EmbeddingBackend``, you must pass the
        same *embed_fn* here (it cannot be serialised).
        """
        data = json.loads(Path(path).read_text())
        config = data.get("config", {})

        backend_data = data.get("backend", {"type": "tfidf"})
        backend = backend_from_dict(backend_data, embed_fn=embed_fn)

        mem = cls(
            max_zettels=config.get("max_zettels", 5000),
            connection_threshold=config.get("connection_threshold", 0.25),
            backend=backend,
        )
        for zd in data.get("zettels", []):
            mem._zettels[zd["id"]] = Zettel.from_dict(zd)
        mem._backend.needs_rebuild = True
        return mem

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _make_id(self, content: str) -> str:
        raw = f"{content}:{time.time_ns()}"
        return hashlib.sha256(raw.encode()).hexdigest()[:16]

    def _rebuild_index_if_needed(self) -> None:
        if not self._backend.needs_rebuild or not self._zettels:
            return
        ids = list(self._zettels.keys())
        texts = [self._zettels[zid].content for zid in ids]
        self._backend.build_index(ids, texts)

    def _link_zettel(self, new_zettel: Zettel) -> None:
        """Find existing zettels similar to the new one and create bidirectional links."""
        similar = self._backend.find_similar(new_zettel.content, self.connection_threshold)
        for zid, _sim in similar:
            if zid in self._zettels:
                new_zettel.connections.add(zid)
                self._zettels[zid].connections.add(new_zettel.id)

    def _keyword_search(self, query: str, limit: int) -> list[SearchResult]:
        """Fallback search when the backend produces no results."""
        query_words = set(query.lower().split())
        results: list[SearchResult] = []

        for zettel in self._zettels.values():
            words = set(zettel.content.lower().split())
            overlap = len(query_words & words)
            if overlap > 0:
                score = overlap / max(len(query_words), 1)
                results.append(SearchResult(zettel=zettel, score=score))

        results.sort(key=lambda r: r.score, reverse=True)
        return results[:limit]

    def _evict(self) -> None:
        """Remove least-valuable zettels when over capacity."""
        now = time.time()
        scored = []
        for zid, z in self._zettels.items():
            recency = 1.0 / (1.0 + (now - z.accessed_at) / 86400)
            value = z.importance * (1 + z.access_count) * recency
            scored.append((zid, value))

        scored.sort(key=lambda x: x[1])
        to_remove = len(self._zettels) - self.max_zettels + 10
        for zid, _ in scored[:to_remove]:
            self.delete(zid)
        self._backend.needs_rebuild = True
