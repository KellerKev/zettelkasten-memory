"""Tests for the core ZettelMemory engine.

All embedding tests use real Ollama embeddings (nomic-embed-text).
"""

import json
from pathlib import Path

import numpy as np
import pytest

from zettelkasten_memory import ZettelMemory, Zettel, SearchResult
from zettelkasten_memory.backends import TfidfBackend, EmbeddingBackend

# ------------------------------------------------------------------
# TF-IDF tests (no embeddings needed)
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
    assert "PostgreSQL" in results[0].zettel.content


def test_search_empty_memory():
    mem = ZettelMemory()
    results = mem.search("anything")
    assert results == []


def test_auto_linking():
    mem = ZettelMemory()
    z1 = mem.add("FastAPI is used for the REST API endpoints")
    z2 = mem.add("The REST API uses JWT authentication on all endpoints")

    mem.search("api")

    z1_fresh = mem.get(z1.id)
    z2_fresh = mem.get(z2.id)
    assert isinstance(z1_fresh.connections, set)
    assert isinstance(z2_fresh.connections, set)


def test_delete():
    mem = ZettelMemory()
    z = mem.add("temporary note")
    assert mem.get(z.id) is not None

    ok = mem.delete(z.id)
    assert ok is True
    assert mem.get(z.id) is None
    assert mem.delete(z.id) is False


def test_delete_cleans_connections():
    mem = ZettelMemory(connection_threshold=0.0)
    z1 = mem.add("note one about APIs")
    z2 = mem.add("note two about APIs")

    mem.search("APIs")

    mem.delete(z1.id)
    z2_fresh = mem.get(z2.id)
    assert z1.id not in z2_fresh.connections


def test_get_connected():
    mem = ZettelMemory(connection_threshold=0.0)
    z1 = mem.add("central note about machine learning")
    z2 = mem.add("related note about machine learning models")
    z3 = mem.add("another note about machine learning training")

    connected = mem.get_connected(z1.id, depth=1)
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


def test_prune_dry_run_then_delete():
    import time

    mem = ZettelMemory()
    keep = mem.add("actively used important note", importance=0.9)
    stale = mem.add("old forgotten trivia", importance=0.1)
    mem._zettels[stale.id].accessed_at = time.time() - 40 * 86400

    # dry run: reports the stale candidate, deletes nothing
    res = mem.prune(max_age_days=30, dry_run=True)
    assert res["dry_run"] is True and res["removed"] == 0
    assert res["matched"] == 1 and res["candidates"][0]["id"] == stale.id
    assert stale.id in mem._zettels

    # real run: deletes the stale one, keeps the fresh important one
    res = mem.prune(max_age_days=30, dry_run=False)
    assert res["removed"] == 1
    assert stale.id not in mem._zettels and keep.id in mem._zettels


def test_prune_is_namespace_scoped():
    mem = ZettelMemory()
    a = mem.add("tenant a note", namespace="a", importance=0.1)
    b = mem.add("tenant b note", namespace="b", importance=0.1)
    # pruning scope "a" (no filters -> everything in a is a candidate) must not
    # touch namespace b
    res = mem.prune(namespace="a", dry_run=False)
    assert a.id not in mem._zettels
    assert b.id in mem._zettels
    assert all(c["namespace"] == "a" for c in res["candidates"])


def test_importance_decay_default_off():
    import time

    mem = ZettelMemory()  # no decay configured
    z = mem.add("note", importance=0.5)
    mem._zettels[z.id].accessed_at = time.time() - 365 * 86400  # a year stale
    assert mem._effective_importance(mem._zettels[z.id], time.time()) == 0.5


def test_importance_decay_lowers_stale_memories():
    import time

    mem = ZettelMemory(importance_half_life_days=7)
    now = time.time()
    fresh = mem.add("fresh note", importance=0.5)
    stale = mem.add("stale note", importance=0.5)
    mem._zettels[stale.id].accessed_at = now - 14 * 86400  # ~2 half-lives -> ~0.25x
    eff_fresh = mem._effective_importance(mem._zettels[fresh.id], now)
    eff_stale = mem._effective_importance(mem._zettels[stale.id], now)
    assert eff_stale < eff_fresh
    assert abs(eff_stale - 0.5 * 0.25) < 0.02


def test_reinforcement_boosts_and_caps():
    mem = ZettelMemory(reinforcement=0.1)
    z = mem.add("reinforce me about databases", importance=0.5)
    mem.search("databases")
    assert mem._zettels[z.id].importance > 0.5
    for _ in range(20):
        mem.search("databases")
    assert mem._zettels[z.id].importance <= 1.0


def test_decay_reinforcement_config_roundtrip(tmp_path):
    mem = ZettelMemory(importance_half_life_days=30, reinforcement=0.05)
    mem.add("note")
    path = tmp_path / "m.json"
    mem.save(path)
    loaded = ZettelMemory.load(path)
    assert loaded.importance_half_life_days == 30
    assert loaded.reinforcement == 0.05


def test_importance_clamping():
    mem = ZettelMemory()
    z1 = mem.add("test", importance=2.0)
    assert z1.importance == 1.0

    z2 = mem.add("test2", importance=-0.5)
    assert z2.importance == 0.0


def test_tags_extraction():
    mem = ZettelMemory()
    z = mem.add("The PostgreSQL database handles authentication and user sessions")
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
# Backend protocol tests
# ------------------------------------------------------------------


def test_tfidf_backend_protocol():
    from zettelkasten_memory.backends import SearchBackend

    assert isinstance(TfidfBackend(), SearchBackend)


def test_embedding_backend_protocol(ollama_embed_fn):
    from zettelkasten_memory.backends import SearchBackend

    assert isinstance(EmbeddingBackend(embed_fn=ollama_embed_fn), SearchBackend)


def test_tfidf_backend_roundtrip():
    b = TfidfBackend(max_features=1000, ngram_range=(1, 3))
    d = b.to_dict()
    b2 = TfidfBackend.from_dict(d)
    assert b2.max_features == 1000
    assert b2.ngram_range == (1, 3)


# ------------------------------------------------------------------
# Hybrid backend tests (deterministic fake embedder — no Ollama needed)
# ------------------------------------------------------------------

_HYBRID_VOCAB = ["fastapi", "rest", "api", "postgres", "database", "docker", "deploy"]


def _fake_embed(texts):
    out = []
    for t in texts:
        low = t.lower()
        v = np.array([1.0 if w in low else 0.0 for w in _HYBRID_VOCAB], dtype=np.float32)
        if v.sum() == 0:
            v[0] = 1e-3
        out.append(v)
    return np.array(out)


def test_hybrid_backend_protocol_and_search():
    from zettelkasten_memory import HybridBackend
    from zettelkasten_memory.backends import SearchBackend

    backend = HybridBackend(embed_fn=_fake_embed)
    assert isinstance(backend, SearchBackend)

    mem = ZettelMemory(backend=backend)
    mem.add("The project uses FastAPI for the REST API")
    mem.add("PostgreSQL is the primary database")
    results = mem.search("database", limit=3)
    assert results and "PostgreSQL" in results[0].zettel.content


def test_hybrid_backend_requires_embedding_source():
    from zettelkasten_memory import HybridBackend

    with pytest.raises(ValueError):
        HybridBackend()


def test_hybrid_backend_roundtrip(tmp_path):
    from zettelkasten_memory import HybridBackend

    mem = ZettelMemory(backend=HybridBackend(embed_fn=_fake_embed))
    mem.add("Deployment uses Docker containers")
    path = tmp_path / "hybrid.json"
    mem.save(path)

    loaded = ZettelMemory.load(path, embed_fn=_fake_embed)
    assert loaded._backend.to_dict()["type"] == "hybrid"
    assert loaded.search("Docker deploy", limit=2)


# ------------------------------------------------------------------
# FAISS backend tests (skipped if faiss-cpu is not installed)
# ------------------------------------------------------------------


@pytest.mark.parametrize("index", ["flat", "hnsw"])
def test_faiss_backend_search_and_roundtrip(index, tmp_path):
    pytest.importorskip("faiss")
    from zettelkasten_memory import FaissBackend
    from zettelkasten_memory.backends import SearchBackend

    backend = FaissBackend(embed_fn=_fake_embed, index=index)
    assert isinstance(backend, SearchBackend)

    mem = ZettelMemory(backend=backend)
    mem.add("The project uses FastAPI for the REST API")
    mem.add("PostgreSQL is the primary database")
    results = mem.search("database", limit=3)
    assert results and "PostgreSQL" in results[0].zettel.content

    # the serialised FAISS index reloads without re-embedding
    path = tmp_path / f"faiss_{index}.json"
    mem.save(path)
    loaded = ZettelMemory.load(path, embed_fn=_fake_embed)
    assert loaded._backend.to_dict()["type"] == "faiss"
    assert loaded.search("FastAPI REST", limit=2)


def test_faiss_backend_rejects_bad_index():
    pytest.importorskip("faiss")
    from zettelkasten_memory import FaissBackend

    with pytest.raises(ValueError):
        FaissBackend(embed_fn=_fake_embed, index="bogus")


# ------------------------------------------------------------------
# Async API (non-blocking wrappers)
# ------------------------------------------------------------------


@pytest.mark.asyncio
async def test_async_add_search_context():
    mem = ZettelMemory()
    z = await mem.aadd("PostgreSQL is the primary database", importance=0.8)
    assert z.id in mem._zettels

    results = await mem.asearch("database", limit=3)
    assert results and "PostgreSQL" in results[0].zettel.content

    ctx = await mem.aget_context("database")
    assert isinstance(ctx, str) and ctx


@pytest.mark.asyncio
async def test_async_writes_are_serialized():
    import asyncio as _asyncio

    mem = ZettelMemory()
    await _asyncio.gather(*(mem.aadd(f"concurrent note number {i}") for i in range(15)))
    # every write landed; the per-store lock prevents lost updates on _zettels
    assert len(mem._zettels) == 15


# ------------------------------------------------------------------
# Embedding backend tests (real Ollama embeddings)
# ------------------------------------------------------------------


def test_embedding_backend_search(ollama_embed_fn):
    """EmbeddingBackend indexes and returns semantically relevant results."""
    backend = EmbeddingBackend(embed_fn=ollama_embed_fn)

    ids = ["a", "b", "c"]
    texts = ["machine learning models", "database schema design", "neural networks"]
    backend.build_index(ids, texts)

    results = backend.query("deep learning")
    assert len(results) > 0
    # "neural networks" or "machine learning" should rank top for "deep learning"
    top_id = results[0][0]
    assert top_id in ("a", "c")


def test_embedding_backend_find_similar(ollama_embed_fn):
    backend = EmbeddingBackend(embed_fn=ollama_embed_fn)

    ids = ["a", "b"]
    texts = [
        "the quick brown fox jumps over the lazy dog",
        "a fast brown fox leaps over a sleepy dog",
    ]
    backend.build_index(ids, texts)

    similar = backend.find_similar("the quick brown fox jumps over the lazy dog", threshold=0.5)
    assert len(similar) == 2  # both should be very similar


def test_zettel_memory_with_embedding_backend(ollama_embed_fn):
    """ZettelMemory works end-to-end with real embeddings."""
    mem = ZettelMemory(backend=EmbeddingBackend(embed_fn=ollama_embed_fn))

    mem.add("The project uses FastAPI for REST APIs")
    mem.add("PostgreSQL is the primary database")
    mem.add("The user prefers dark mode")

    results = mem.search("database")
    assert len(results) > 0
    assert "PostgreSQL" in results[0].zettel.content


def test_embedding_search_semantic_understanding(ollama_embed_fn):
    """Real embeddings understand synonyms and paraphrases."""
    mem = ZettelMemory(backend=EmbeddingBackend(embed_fn=ollama_embed_fn))

    mem.add("The application is deployed using Docker containers on AWS ECS")
    mem.add("Authentication uses JSON Web Tokens for session management")
    mem.add("The frontend is built with React and TypeScript")

    # Query with different words — should still find Docker/deployment
    results = mem.search("how do we ship the app to production?")
    assert len(results) > 0
    assert "Docker" in results[0].zettel.content


# ------------------------------------------------------------------
# Save/load with embedding backend
# ------------------------------------------------------------------


def test_save_load_with_embedding_backend(tmp_path, ollama_embed_fn):
    mem = ZettelMemory(backend=EmbeddingBackend(embed_fn=ollama_embed_fn))
    mem.add("some content about databases")
    path = tmp_path / "emb_memory.json"
    mem.save(path)

    loaded = ZettelMemory.load(path, embed_fn=ollama_embed_fn)
    assert isinstance(loaded._backend, EmbeddingBackend)
    assert len(loaded._zettels) == 1


def test_save_load_preserves_tfidf_backend(tmp_path):
    mem = ZettelMemory()
    mem.add("hello world")
    path = tmp_path / "tfidf_memory.json"
    mem.save(path)

    loaded = ZettelMemory.load(path)
    assert isinstance(loaded._backend, TfidfBackend)


def test_backend_config_in_saved_json(tmp_path):
    mem = ZettelMemory()
    mem.add("test")
    path = tmp_path / "mem.json"
    mem.save(path)

    data = json.loads(path.read_text())
    assert "backend" in data
    assert data["backend"]["type"] == "tfidf"


# ------------------------------------------------------------------
# Embedding persistence tests
# ------------------------------------------------------------------


def test_save_load_preserves_embedding_vectors(tmp_path, ollama_embed_fn):
    """Vectors survive a save/load roundtrip — no re-embedding needed on load."""
    mem = ZettelMemory(backend=EmbeddingBackend(embed_fn=ollama_embed_fn))

    mem.add("machine learning is a branch of artificial intelligence")
    mem.add("relational databases store data in tables")
    mem.add("REST APIs use HTTP methods for CRUD operations")

    # Force index build
    mem.search("test query")

    path = tmp_path / "persist.json"
    mem.save(path)

    loaded = ZettelMemory.load(path, embed_fn=ollama_embed_fn)
    assert isinstance(loaded._backend, EmbeddingBackend)
    assert len(loaded._zettels) == 3
    assert not loaded._backend.needs_rebuild

    # Search should still return semantically correct results
    results = loaded.search("database")
    assert len(results) > 0
    assert "database" in results[0].zettel.content.lower()


def test_save_load_embedding_without_embed_fn(tmp_path, ollama_embed_fn):
    """Loading with persisted vectors works without embed_fn for read-only access."""
    mem = ZettelMemory(backend=EmbeddingBackend(embed_fn=ollama_embed_fn))

    mem.add("test content one")
    mem.add("test content two")
    mem.search("trigger build")

    path = tmp_path / "readonly.json"
    mem.save(path)

    loaded = ZettelMemory.load(path)
    assert isinstance(loaded._backend, EmbeddingBackend)
    assert len(loaded._zettels) == 2
    assert not loaded._backend.needs_rebuild


def test_save_load_backward_compat_no_vectors(tmp_path, ollama_embed_fn):
    """Old save files without vectors trigger re-embedding on first search."""
    mem = ZettelMemory(backend=EmbeddingBackend(embed_fn=ollama_embed_fn))
    mem.add("some content about APIs")

    path = tmp_path / "old_format.json"
    mem.save(path)

    # Simulate old format by stripping vectors from saved JSON
    data = json.loads(path.read_text())
    data["backend"] = {"type": "embedding", "batch_size": 64}
    path.write_text(json.dumps(data))

    loaded = ZettelMemory.load(path, embed_fn=ollama_embed_fn)
    assert loaded._backend.needs_rebuild
    results = loaded.search("APIs")
    assert isinstance(results, list)
