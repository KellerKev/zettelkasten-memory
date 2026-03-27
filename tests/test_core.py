"""Tests for the core ZettelMemory engine."""

import json
import tempfile
from pathlib import Path

import numpy as np

from zettelkasten_memory import ZettelMemory, Zettel, SearchResult
from zettelkasten_memory.backends import TfidfBackend, EmbeddingBackend


# ------------------------------------------------------------------
# Original tests (TF-IDF, backward compat)
# ------------------------------------------------------------------


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


# ------------------------------------------------------------------
# Backend-specific tests
# ------------------------------------------------------------------


def test_tfidf_backend_protocol():
    """TfidfBackend satisfies the SearchBackend protocol."""
    from zettelkasten_memory.backends import SearchBackend

    assert isinstance(TfidfBackend(), SearchBackend)


def test_embedding_backend_protocol():
    """EmbeddingBackend satisfies the SearchBackend protocol."""
    from zettelkasten_memory.backends import SearchBackend

    dummy_fn = lambda texts: np.random.randn(len(texts), 8)
    assert isinstance(EmbeddingBackend(embed_fn=dummy_fn), SearchBackend)


def test_tfidf_backend_roundtrip():
    """TfidfBackend serialises and deserialises cleanly."""
    b = TfidfBackend(max_features=1000, ngram_range=(1, 3))
    d = b.to_dict()
    b2 = TfidfBackend.from_dict(d)
    assert b2.max_features == 1000
    assert b2.ngram_range == (1, 3)


def _make_deterministic_embed_fn(dim: int = 32):
    """Return an embed_fn that produces deterministic vectors from text."""
    def embed(texts: list[str]) -> np.ndarray:
        vecs = []
        for t in texts:
            rng = np.random.RandomState(hash(t) % 2**31)
            vecs.append(rng.randn(dim))
        return np.array(vecs)
    return embed


def test_embedding_backend_search():
    """EmbeddingBackend can index and query."""
    embed_fn = _make_deterministic_embed_fn()
    backend = EmbeddingBackend(embed_fn=embed_fn)

    ids = ["a", "b", "c"]
    texts = ["machine learning models", "database schema", "neural networks"]
    backend.build_index(ids, texts)

    results = backend.query("deep learning")
    assert isinstance(results, list)
    # Should return some results (exact ranking depends on random vectors)


def test_embedding_backend_find_similar():
    embed_fn = _make_deterministic_embed_fn()
    backend = EmbeddingBackend(embed_fn=embed_fn)

    ids = ["a", "b"]
    texts = ["the quick brown fox", "the quick brown fox jumps"]
    backend.build_index(ids, texts)

    similar = backend.find_similar("the quick brown fox", threshold=-1.0)
    assert len(similar) > 0


def test_zettel_memory_with_embedding_backend():
    """ZettelMemory works end-to-end with EmbeddingBackend."""
    embed_fn = _make_deterministic_embed_fn()
    mem = ZettelMemory(backend=EmbeddingBackend(embed_fn=embed_fn))

    mem.add("The project uses FastAPI for REST APIs")
    mem.add("PostgreSQL is the primary database")
    mem.add("The user prefers dark mode")

    results = mem.search("database")
    assert isinstance(results, list)


def test_save_load_with_embedding_backend(tmp_path):
    """Save/load round-trip preserves backend type."""
    embed_fn = _make_deterministic_embed_fn()
    mem = ZettelMemory(backend=EmbeddingBackend(embed_fn=embed_fn))
    mem.add("some content")
    path = tmp_path / "emb_memory.json"
    mem.save(path)

    # Must pass embed_fn when loading an embedding-backed memory
    loaded = ZettelMemory.load(path, embed_fn=embed_fn)
    assert isinstance(loaded._backend, EmbeddingBackend)
    assert len(loaded._zettels) == 1


def test_save_load_preserves_tfidf_backend(tmp_path):
    """Default save/load still uses TfidfBackend."""
    mem = ZettelMemory()
    mem.add("hello world")
    path = tmp_path / "tfidf_memory.json"
    mem.save(path)

    loaded = ZettelMemory.load(path)
    assert isinstance(loaded._backend, TfidfBackend)


def test_backend_config_in_saved_json(tmp_path):
    """Saved JSON includes backend config."""
    mem = ZettelMemory()
    mem.add("test")
    path = tmp_path / "mem.json"
    mem.save(path)

    data = json.loads(path.read_text())
    assert "backend" in data
    assert data["backend"]["type"] == "tfidf"
