"""
CrewAI adapter — plug ZettelMemory into CrewAI's memory system.

CrewAI memory works through Storage backends. This module provides
ZettelStorage which implements CrewAI's RAGStorage interface so you
can use Zettelkasten memory as the backing store for any CrewAI
memory type (short-term, long-term, entity).

Usage:
    from crewai import Crew, Agent, Task
    from crewai.memory import ShortTermMemory
    from zettelkasten_memory.adapters.crewai import ZettelStorage

    storage = ZettelStorage(persist_path="crew_memory.json")

    crew = Crew(
        agents=[...],
        tasks=[...],
        memory=True,
        short_term_memory=ShortTermMemory(storage=storage),
    )
"""

from __future__ import annotations

from typing import Any

from zettelkasten_memory.core import ZettelMemory


class ZettelStorage:
    """
    CrewAI-compatible storage backend backed by ZettelMemory.

    Implements the interface that CrewAI's memory classes expect from a
    storage object: save(), search(), and reset().
    """

    def __init__(
        self,
        persist_path: str | None = None,
        max_zettels: int = 5000,
        **kwargs: Any,
    ):
        self._persist_path = persist_path
        self._mem = ZettelMemory(max_zettels=max_zettels)

        if persist_path:
            try:
                self._mem = ZettelMemory.load(persist_path)
            except (FileNotFoundError, Exception):
                pass  # start fresh

    def save(self, value: Any, metadata: dict[str, Any] | None = None, agent: str = "") -> None:
        """Store a memory entry. Called by CrewAI after task execution."""
        if isinstance(value, dict):
            content = value.get("content", value.get("text", str(value)))
        else:
            content = str(value)

        tags: set[str] = set()
        if agent:
            tags.add(f"agent:{agent}")

        self._mem.add(content, metadata=metadata or {}, tags=tags)

        if self._persist_path:
            self._mem.save(self._persist_path)

    def search(self, query: str, limit: int = 3, score_threshold: float = 0.0) -> list[dict]:
        """Search memory. Called by CrewAI before task execution for context."""
        results = self._mem.search(query, limit=limit, min_score=score_threshold)
        return [
            {
                "context": r.zettel.content,
                "metadata": r.zettel.metadata,
                "score": r.score,
            }
            for r in results
        ]

    def reset(self) -> None:
        """Clear all memory."""
        self._mem = ZettelMemory(max_zettels=self._mem.max_zettels)
        if self._persist_path:
            self._mem.save(self._persist_path)
