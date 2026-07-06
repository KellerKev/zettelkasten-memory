"""Consumer API contract tests.

These reproduce the exact call patterns of the two downstream consumers of this
library (a terminal agent plugin and a knowledge-agent memory plugin), both of
which import `zettelkasten_memory` directly.  Every test here must keep passing
across refactors: a failure means a downstream project breaks on its next
dependency solve.

Contract points covered (do not weaken):
- top-level exports: ZettelMemory, EmbeddingBackend, Zettel, SearchResult
- ZettelMemory(max_zettels=, connection_threshold=, backend=), backend=None -> TF-IDF
- add(content, tags=set|None, importance=float) with POSITIONAL content
- search(query, limit=) with POSITIONAL query -> objects with .zettel/.score
- get_context(query, max_tokens=) -> str
- delete(id) -> bool
- save(str_path) / ZettelMemory.load(path, embed_fn=) classmethod
- stats as a PROPERTY with total_zettels/total_connections/avg_connections
- Zettel fields: id, content, tags, importance, connections (mutable set),
  accessed_at, access_count
- private-but-load-bearing: mem._zettels dict, EmbeddingBackend._embed_fn,
  EmbeddingBackend(embed_fn=), EmbeddingBackend.from_provider
- on-disk v1 JSON files load unchanged
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from zettelkasten_memory import EmbeddingBackend, SearchResult, Zettel, ZettelMemory


def _fake_embed(texts):
    rng = np.random.default_rng(42)
    out = []
    for t in texts:
        state = np.random.default_rng(abs(hash(t)) % (2**32))
        out.append(state.random(8))
    return np.array(out)


def test_constructor_kwargs_and_tfidf_default():
    mem = ZettelMemory(max_zettels=2000, connection_threshold=0.25)
    assert mem.max_zettels == 2000
    mem2 = ZettelMemory(max_zettels=100, connection_threshold=0.3, backend=None)
    assert mem2 is not None


def test_add_positional_content_returns_zettel():
    mem = ZettelMemory(max_zettels=100, connection_threshold=0.25)
    z = mem.add("the user prefers dark mode", tags={"preference"}, importance=0.8)
    assert isinstance(z, Zettel)
    assert isinstance(z.id, str)
    assert z.content == "the user prefers dark mode"
    assert isinstance(z.tags, set) and "preference" in z.tags
    assert z.importance == 0.8
    assert isinstance(z.connections, set)
    # tags=None accepted
    z2 = mem.add("another note", tags=None, importance=0.5)
    assert isinstance(z2, Zettel)


def test_search_positional_query_result_shape():
    mem = ZettelMemory(max_zettels=100, connection_threshold=0.25)
    mem.add("the project uses FastAPI and PostgreSQL")
    mem.add("deployment runs on kubernetes")
    results = mem.search("what database does the project use", limit=5)
    assert isinstance(results, list)
    for r in results:
        assert isinstance(r.zettel, Zettel)
        assert isinstance(r.score, float)
        assert isinstance(r.zettel.tags, set)
        assert isinstance(r.zettel.connections, set)


def test_get_context_keyword_max_tokens():
    mem = ZettelMemory(max_zettels=100, connection_threshold=0.25)
    mem.add("kubernetes cluster runs three nodes")
    ctx = mem.get_context("kubernetes nodes", max_tokens=2000)
    assert isinstance(ctx, str)


def test_delete_returns_bool():
    mem = ZettelMemory(max_zettels=100, connection_threshold=0.25)
    z = mem.add("temporary note")
    assert mem.delete(z.id) is True
    assert mem.delete("nonexistent") is False


def test_stats_is_property_with_required_keys():
    mem = ZettelMemory(max_zettels=100, connection_threshold=0.25)
    mem.add("a note about testing")
    s = mem.stats  # property access, not a call
    assert isinstance(s, dict)
    for key in ("total_zettels", "total_connections", "avg_connections"):
        assert key in s
    # consumers mutate the returned dict
    s["plugin"] = "zettel"


def test_private_zettels_dict_contract():
    mem = ZettelMemory(max_zettels=100, connection_threshold=0.25)
    z = mem.add("indexed content")
    assert isinstance(mem._zettels, dict)
    assert mem._zettels.get(z.id) is z
    assert len(mem._zettels) == 1
    values = sorted(mem._zettels.values(), key=lambda x: x.accessed_at)
    assert values[0].access_count == 0


def test_connections_mutable_set_direct_mutation():
    mem = ZettelMemory(max_zettels=100, connection_threshold=0.25)
    a = mem.add("skill: reading policies")
    b = mem.add("knowledge: policy database layout")
    # skill_indexer-style direct bidirectional edge building
    a.connections.add(b.id)
    b.connections.add(a.id)
    assert b.id in mem._zettels[a.id].connections


def test_save_load_roundtrip_str_path(tmp_path):
    mem = ZettelMemory(max_zettels=100, connection_threshold=0.25)
    z = mem.add("persisted note", tags={"keep"}, importance=0.7)
    path = tmp_path / "memory.json"
    mem.save(str(path))
    loaded = ZettelMemory.load(str(path))
    assert z.id in loaded._zettels
    got = loaded._zettels[z.id]
    assert got.content == "persisted note"
    assert got.tags >= {"keep"}
    assert got.importance == 0.7


def test_load_with_embed_fn_kwarg(tmp_path):
    backend = EmbeddingBackend(embed_fn=_fake_embed)
    mem = ZettelMemory(max_zettels=100, connection_threshold=0.25, backend=backend)
    mem.add("embedded note")
    path = tmp_path / "emb.json"
    mem.save(str(path))
    loaded = ZettelMemory.load(str(path), embed_fn=_fake_embed)
    assert loaded.search("embedded note", limit=3)


def test_embedding_backend_private_embed_fn_attr():
    backend = EmbeddingBackend(embed_fn=_fake_embed)
    # both consumers read this private attribute to re-feed load()
    assert backend._embed_fn is _fake_embed
    assert getattr(backend, "_embed_fn", None) is _fake_embed


def test_embedding_backend_from_provider_classmethod_exists():
    assert callable(EmbeddingBackend.from_provider)


def test_legacy_v1_file_loads(tmp_path):
    v1 = {
        "version": 1,
        "zettels": [
            {
                "id": "abc123",
                "content": "legacy note",
                "metadata": {},
                "tags": ["old"],
                "connections": [],
                "created_at": 1700000000.0,
                "accessed_at": 1700000000.0,
                "access_count": 3,
                "importance": 0.6,
            }
        ],
        "config": {"max_zettels": 5000, "connection_threshold": 0.25},
        "backend": {"type": "tfidf"},
    }
    path = tmp_path / "legacy.json"
    path.write_text(json.dumps(v1))
    mem = ZettelMemory.load(str(path))
    z = mem._zettels["abc123"]
    assert z.content == "legacy note"
    assert z.access_count == 3
    assert isinstance(z.tags, set) and isinstance(z.connections, set)
    assert isinstance(mem.search("legacy", limit=5), list)


def test_search_result_dataclass_public():
    r = SearchResult(zettel=Zettel(id="x", content="y"), score=0.5)
    assert r.score == 0.5
