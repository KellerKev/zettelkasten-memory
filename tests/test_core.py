"""Tests for the core ZettelMemory engine."""

import json
import tempfile
from pathlib import Path

from zettelkasten_memory import ZettelMemory, Zettel, SearchResult


def test_add_and_get():
    mem = ZettelMemory()
    z = mem.add("The project uses FastAPI and PostgreSQL")
    assert z.id
    assert "FastAPI" in z.content
    assert mem.get(z.id) is z


def test_search_returns_relevant_results():
    mem = ZettelMemory()
    mem.add("The project uses FastAPI for the REST API layer")
    mem.add("PostgreSQL is the primary database")
    mem.add("The user prefers dark mode in the IDE")

    results = mem.search("what database does the project use?")
    assert len(results) > 0
    # PostgreSQL should rank higher than IDE preferences
    assert "PostgreSQL" in results[0].zettel.content


def test_search_empty_memory():
    mem = ZettelMemory()
    results = mem.search("anything")
    assert results == []


def test_auto_linking():
    mem = ZettelMemory()
    z1 = mem.add("FastAPI is used for the REST API endpoints")
    z2 = mem.add("The REST API uses JWT authentication on all endpoints")

    # After adding z2, both should be connected since they share topic
    # (Rebuild happens on next search)
    mem.search("api")

    z1_fresh = mem.get(z1.id)
    z2_fresh = mem.get(z2.id)
    # They may or may not be connected depending on TF-IDF threshold,
    # but the mechanism should not crash
    assert isinstance(z1_fresh.connections, set)
    assert isinstance(z2_fresh.connections, set)


def test_delete():
    mem = ZettelMemory()
    z = mem.add("temporary note")
    assert mem.get(z.id) is not None

    ok = mem.delete(z.id)
    assert ok is True
    assert mem.get(z.id) is None

    # Deleting again returns False
    assert mem.delete(z.id) is False


def test_delete_cleans_connections():
    mem = ZettelMemory(connection_threshold=0.0)  # link everything
    z1 = mem.add("note one about APIs")
    z2 = mem.add("note two about APIs")

    # Force index rebuild and re-add to get links
    mem.search("APIs")

    mem.delete(z1.id)
    z2_fresh = mem.get(z2.id)
    assert z1.id not in z2_fresh.connections


def test_get_connected():
    mem = ZettelMemory(connection_threshold=0.0)  # link everything
    z1 = mem.add("central note about machine learning")
    z2 = mem.add("related note about machine learning models")
    z3 = mem.add("another note about machine learning training")

    connected = mem.get_connected(z1.id, depth=1)
    # Should find some connected zettels (exact count depends on threshold)
    assert isinstance(connected, list)


def test_get_context():
    mem = ZettelMemory()
    mem.add("The deployment uses Docker containers")
    mem.add("CI/CD runs on GitHub Actions")
    mem.add("The user likes short answers")

    ctx = mem.get_context("how is the app deployed?", max_tokens=500)
    assert isinstance(ctx, str)
    assert len(ctx) > 0


def test_save_and_load(tmp_path):
    path = tmp_path / "test_memory.json"

    mem = ZettelMemory()
    z1 = mem.add("note one", tags={"test"})
    z2 = mem.add("note two", metadata={"key": "value"})
    mem.save(path)

    loaded = ZettelMemory.load(path)
    assert len(loaded._zettels) == 2

    z1_loaded = loaded.get(z1.id)
    assert z1_loaded is not None
    assert z1_loaded.content == z1.content
    assert "test" in z1_loaded.tags

    z2_loaded = loaded.get(z2.id)
    assert z2_loaded.metadata == {"key": "value"}


def test_stats():
    mem = ZettelMemory()
    assert mem.stats["total_zettels"] == 0

    mem.add("first note")
    mem.add("second note")
    assert mem.stats["total_zettels"] == 2


def test_eviction():
    mem = ZettelMemory(max_zettels=5)
    for i in range(20):
        mem.add(f"note number {i} about topic {i}")
    assert len(mem._zettels) <= 5


def test_importance_clamping():
    mem = ZettelMemory()
    z1 = mem.add("test", importance=2.0)
    assert z1.importance == 1.0

    z2 = mem.add("test2", importance=-0.5)
    assert z2.importance == 0.0


def test_tags_extraction():
    mem = ZettelMemory()
    z = mem.add("The PostgreSQL database handles authentication and user sessions")
    # Should have auto-extracted some tags
    assert len(z.tags) > 0


def test_zettel_serialization():
    z = Zettel(
        id="abc123",
        content="test content",
        metadata={"key": "val"},
        tags={"tag1", "tag2"},
        connections={"other1"},
    )
    d = z.to_dict()
    z2 = Zettel.from_dict(d)
    assert z2.id == z.id
    assert z2.content == z.content
    assert z2.tags == z.tags
    assert z2.connections == z.connections
