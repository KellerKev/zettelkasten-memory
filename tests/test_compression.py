"""Tests for the TurboQuant compression layer.

Uses real Ollama embeddings for integration tests.
"""

import numpy as np
import pytest

from zettelkasten_memory.compression import CompressedVectors, TurboQuantCompressor
from zettelkasten_memory.backends import EmbeddingBackend
from zettelkasten_memory import ZettelMemory


# ------------------------------------------------------------------
# Compress / decompress roundtrip
# ------------------------------------------------------------------


def test_compress_returns_compressed_vectors(ollama_embed_fn, ollama_dim):
    """Compress real embeddings and verify metadata."""
    vecs = ollama_embed_fn(["hello world", "machine learning", "database schema"])
    compressor = TurboQuantCompressor(seed=123)
    compressed = compressor.compress(vecs)

    assert isinstance(compressed, CompressedVectors)
    assert compressed.n_vectors == 3
    assert compressed.orig_dim == ollama_dim


def test_compress_decompress_serialization_roundtrip(ollama_embed_fn):
    """CompressedVectors survive to_dict / from_dict with real embeddings."""
    vecs = ollama_embed_fn(["alpha beta gamma", "delta epsilon zeta"])
    compressor = TurboQuantCompressor()
    compressed = compressor.compress(vecs)

    data = compressed.to_dict()
    restored = CompressedVectors.from_dict(data)

    assert restored.n_vectors == compressed.n_vectors
    assert restored.orig_dim == compressed.orig_dim
    np.testing.assert_array_equal(restored.codes, compressed.codes)
    np.testing.assert_array_equal(restored.mins, compressed.mins)
    np.testing.assert_array_equal(restored.scales, compressed.scales)
    np.testing.assert_array_equal(restored.residual_bits, compressed.residual_bits)


# ------------------------------------------------------------------
# Asymmetric search accuracy with real embeddings
# ------------------------------------------------------------------


def test_asymmetric_search_accuracy(ollama_embed_fn):
    """Compressed search preserves ranking of real embeddings."""
    texts = [
        "machine learning algorithms",
        "PostgreSQL database administration",
        "neural network architectures",
        "REST API design patterns",
        "natural language processing",
        "Docker container orchestration",
        "deep learning frameworks",
        "cloud infrastructure management",
        "computer vision models",
        "microservices communication",
        "gradient descent optimization",
        "SQL query performance tuning",
        "transformer attention mechanisms",
        "Kubernetes deployment strategies",
        "recurrent neural networks",
        "load balancing techniques",
        "convolutional neural networks",
        "message queue systems",
        "reinforcement learning agents",
        "CI/CD pipeline automation",
    ]
    vecs = ollama_embed_fn(texts)

    # L2-normalize (same as EmbeddingBackend does)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)
    vecs = vecs / norms

    compressor = TurboQuantCompressor(proj_dim=128, seed=99)
    compressed = compressor.compress(vecs)

    # Query: "deep learning" — should rank ML-related texts highest
    query_vec = ollama_embed_fn(["deep learning"])
    query_vec = query_vec / np.linalg.norm(query_vec)
    query = query_vec.flatten()

    # Brute-force exact
    exact_scores = vecs @ query
    exact_top5 = set(np.argsort(exact_scores)[-5:])

    # Compressed approximate
    approx_scores = compressor.asymmetric_search(query, compressed)
    approx_top5 = set(np.argsort(approx_scores)[-5:])

    # At least 3 of top-5 should overlap
    overlap = len(exact_top5 & approx_top5)
    assert overlap >= 3, f"Only {overlap}/5 overlap between exact and compressed top-5"


# ------------------------------------------------------------------
# Compression ratio with real embeddings
# ------------------------------------------------------------------


def test_compression_ratio(ollama_embed_fn, ollama_dim):
    """Compressed representation should be significantly smaller than float32."""
    # Embed enough texts to measure ratio
    texts = [f"sample text number {i} about topic {i}" for i in range(50)]
    vecs = ollama_embed_fn(texts)

    compressor = TurboQuantCompressor()
    compressed = compressor.compress(vecs)

    original_bytes = vecs.shape[0] * vecs.shape[1] * 4  # float32
    compressed_bytes = (
        compressed.codes.nbytes
        + compressed.mins.nbytes
        + compressed.scales.nbytes
        + compressed.residual_bits.nbytes
    )

    ratio = original_bytes / compressed_bytes
    assert ratio >= 2.5, f"Compression ratio {ratio:.1f}x, expected >= 2.5x"


# ------------------------------------------------------------------
# Integration: EmbeddingBackend + TurboQuantCompressor
# ------------------------------------------------------------------


def test_embedding_backend_with_compressor(ollama_embed_fn):
    """End-to-end: add/search with compression enabled using real embeddings."""
    compressor = TurboQuantCompressor(proj_dim=64, seed=7)
    backend = EmbeddingBackend(embed_fn=ollama_embed_fn, compressor=compressor)

    ids = ["a", "b", "c", "d"]
    texts = [
        "machine learning models and algorithms",
        "relational database schema design",
        "neural network training procedures",
        "REST API endpoint design",
    ]
    backend.build_index(ids, texts)

    results = backend.query("deep learning architectures")
    assert len(results) > 0
    result_ids = {r[0] for r in results}
    assert result_ids.issubset(set(ids))


def test_zettel_memory_with_compressor_semantic(ollama_embed_fn):
    """Full ZettelMemory with compression still returns semantically correct results."""
    compressor = TurboQuantCompressor(proj_dim=64, seed=42)
    backend = EmbeddingBackend(embed_fn=ollama_embed_fn, compressor=compressor)
    mem = ZettelMemory(backend=backend)

    mem.add("The application uses PostgreSQL as its primary relational database")
    mem.add("Authentication is handled via JWT tokens in HTTP headers")
    mem.add("The frontend is a React single-page application")

    results = mem.search("what database does the project use?")
    assert len(results) > 0
    assert "PostgreSQL" in results[0].zettel.content


def test_embedding_backend_compressed_persistence(tmp_path, ollama_embed_fn):
    """Compressed vectors survive save/load via to_dict/from_dict."""
    compressor = TurboQuantCompressor(proj_dim=32, seed=5)
    backend = EmbeddingBackend(embed_fn=ollama_embed_fn, compressor=compressor)

    ids = ["x", "y", "z"]
    texts = ["cats and dogs", "fish and birds", "trees and flowers"]
    backend.build_index(ids, texts)

    data = backend.to_dict()
    assert "compressed_vectors" in data
    assert "compressor" in data

    restored = EmbeddingBackend.from_dict(data, embed_fn=ollama_embed_fn)
    assert not restored.needs_rebuild
    assert len(restored._id_order) == 3

    results = restored.query("animals")
    assert isinstance(results, list)
    assert len(results) > 0


def test_embedding_backend_compressed_readonly_load(ollama_embed_fn):
    """Compressed backend can be loaded without embed_fn for structure access."""
    compressor = TurboQuantCompressor(proj_dim=32, seed=5)
    backend = EmbeddingBackend(embed_fn=ollama_embed_fn, compressor=compressor)

    ids = ["a", "b"]
    texts = ["hello world", "foo bar"]
    backend.build_index(ids, texts)

    data = backend.to_dict()

    restored = EmbeddingBackend.from_dict(data, embed_fn=None)
    assert not restored.needs_rebuild
    assert len(restored._id_order) == 2


def test_compressor_config_roundtrip():
    """TurboQuantCompressor config serialises cleanly."""
    compressor = TurboQuantCompressor(n_bits=3, proj_dim=128, seed=77)
    data = compressor.to_dict()
    restored = TurboQuantCompressor.from_dict(data)
    assert restored.n_bits == 3
    assert restored.proj_dim == 128
    assert restored.seed == 77
