"""Tests for embedding providers.

Tests the Ollama provider end-to-end with real embeddings.
Other providers (OpenAI, Cohere, Voyage, SentenceTransformers) are tested
for construction and registry only — they require API keys or heavy deps.
"""

import numpy as np
import pytest

from zettelkasten_memory.providers import (
    PROVIDER_REGISTRY,
    OllamaEmbeddings,
    OpenAIEmbeddings,
    CohereEmbeddings,
    VoyageEmbeddings,
    SentenceTransformerEmbeddings,
    SnowflakeCortexEmbeddings,
    get_provider,
)
from zettelkasten_memory.backends import EmbeddingBackend


# ------------------------------------------------------------------
# Ollama (real embeddings)
# ------------------------------------------------------------------


def test_ollama_embeddings_shape():
    """OllamaEmbeddings returns correct shape from real Ollama server."""
    embed = OllamaEmbeddings(model="nomic-embed-text")
    result = embed(["hello world", "machine learning"])
    assert result.shape[0] == 2
    assert result.shape[1] > 0
    assert result.dtype == np.float32


def test_ollama_embeddings_deterministic():
    """Same input produces the same embedding."""
    embed = OllamaEmbeddings(model="nomic-embed-text")
    r1 = embed(["test sentence"])
    r2 = embed(["test sentence"])
    np.testing.assert_array_almost_equal(r1, r2, decimal=5)


def test_ollama_embeddings_semantic_similarity():
    """Similar texts produce similar embeddings, dissimilar texts don't."""
    embed = OllamaEmbeddings(model="nomic-embed-text")
    vecs = embed([
        "the cat sat on the mat",          # 0
        "a kitten rested on the rug",       # 1 — similar to 0
        "quantum physics equations",        # 2 — unrelated
    ])
    # Normalize for cosine similarity
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    vecs = vecs / norms

    sim_related = float(vecs[0] @ vecs[1])
    sim_unrelated = float(vecs[0] @ vecs[2])

    assert sim_related > sim_unrelated, (
        f"Related similarity ({sim_related:.3f}) should exceed "
        f"unrelated similarity ({sim_unrelated:.3f})"
    )


def test_ollama_batching():
    """Ollama provider handles batching correctly."""
    embed = OllamaEmbeddings(model="nomic-embed-text", batch_size=2)
    texts = ["text one", "text two", "text three", "text four", "text five"]
    result = embed(texts)
    assert result.shape[0] == 5


def test_ollama_single_text():
    """Single text embedding works."""
    embed = OllamaEmbeddings(model="nomic-embed-text")
    result = embed(["just one sentence"])
    assert result.shape[0] == 1


# ------------------------------------------------------------------
# Provider registry
# ------------------------------------------------------------------


def test_get_provider_ollama():
    provider = get_provider("ollama")
    assert isinstance(provider, OllamaEmbeddings)
    assert callable(provider)


def test_get_provider_unknown():
    with pytest.raises(ValueError, match="Unknown provider"):
        get_provider("nonexistent")


def test_get_provider_case_insensitive():
    provider = get_provider("Ollama")
    assert isinstance(provider, OllamaEmbeddings)


def test_registry_contains_all_providers():
    expected = {"openai", "cohere", "voyage", "sentence-transformers", "local", "ollama", "snowflake", "cortex"}
    assert expected == set(PROVIDER_REGISTRY.keys())


# ------------------------------------------------------------------
# EmbeddingBackend.from_provider()
# ------------------------------------------------------------------


def test_from_provider_ollama():
    """from_provider('ollama') creates a working backend."""
    backend = EmbeddingBackend.from_provider("ollama")
    assert isinstance(backend, EmbeddingBackend)
    assert isinstance(backend._embed_fn, OllamaEmbeddings)


def test_from_provider_ollama_index_and_search():
    """Full pipeline: from_provider → build_index → query with real embeddings."""
    backend = EmbeddingBackend.from_provider("ollama", model="nomic-embed-text")

    ids = ["a", "b", "c"]
    texts = [
        "Python is a programming language",
        "The Eiffel Tower is in Paris",
        "Java and Python are popular languages",
    ]
    backend.build_index(ids, texts)

    results = backend.query("programming languages")
    assert len(results) > 0
    # "Python" and "Java/Python" should rank above Eiffel Tower
    top_ids = [r[0] for r in results[:2]]
    assert "b" not in top_ids, "Eiffel Tower should not be in top 2 for 'programming languages'"


def test_from_provider_with_compressor():
    from zettelkasten_memory.compression import TurboQuantCompressor

    compressor = TurboQuantCompressor()
    backend = EmbeddingBackend.from_provider("ollama", compressor=compressor)
    assert backend._compressor is compressor


# ------------------------------------------------------------------
# Other providers — construction only (no API keys available)
# ------------------------------------------------------------------


def test_openai_provider_constructs():
    provider = OpenAIEmbeddings(api_key="not-a-real-key")
    assert provider.model == "text-embedding-3-small"
    assert callable(provider)


def test_cohere_provider_constructs():
    provider = CohereEmbeddings(api_key="not-a-real-key")
    assert provider.model == "embed-english-v3.0"
    assert callable(provider)


def test_voyage_provider_constructs():
    provider = VoyageEmbeddings(api_key="not-a-real-key")
    assert provider.model == "voyage-3"
    assert callable(provider)


def test_sentence_transformer_provider_constructs():
    provider = SentenceTransformerEmbeddings()
    assert provider.model_name == "all-MiniLM-L6-v2"
    assert callable(provider)


def test_snowflake_provider_constructs():
    provider = SnowflakeCortexEmbeddings(account="org-acct", token="pat-secret")
    assert provider.model == "snowflake-arctic-embed-m-v1.5"
    assert provider._account == "org-acct"
    assert provider._token == "pat-secret"
    assert callable(provider)


def test_snowflake_provider_reads_env():
    import os
    old_acct = os.environ.get("SNOWFLAKE_ACCOUNT")
    old_tok = os.environ.get("SNOWFLAKE_PAT_TOKEN")
    try:
        os.environ["SNOWFLAKE_ACCOUNT"] = "env-org-acct"
        os.environ["SNOWFLAKE_PAT_TOKEN"] = "env-pat-token"
        provider = SnowflakeCortexEmbeddings()
        assert provider._account == "env-org-acct"
        assert provider._token == "env-pat-token"
    finally:
        if old_acct is None:
            os.environ.pop("SNOWFLAKE_ACCOUNT", None)
        else:
            os.environ["SNOWFLAKE_ACCOUNT"] = old_acct
        if old_tok is None:
            os.environ.pop("SNOWFLAKE_PAT_TOKEN", None)
        else:
            os.environ["SNOWFLAKE_PAT_TOKEN"] = old_tok


def test_snowflake_raises_without_account():
    provider = SnowflakeCortexEmbeddings(account=None, token="pat-secret")
    provider._account = None  # force no account
    with pytest.raises(ValueError, match="SNOWFLAKE_ACCOUNT"):
        provider._embed_batch(["test"])


def test_snowflake_raises_without_token():
    provider = SnowflakeCortexEmbeddings(account="org-acct", token=None)
    provider._token = None  # force no token
    with pytest.raises(ValueError, match="SNOWFLAKE_PAT_TOKEN"):
        provider._embed_batch(["test"])


def test_get_provider_snowflake_aliases():
    """Both 'snowflake' and 'cortex' resolve to SnowflakeCortexEmbeddings."""
    p1 = get_provider("snowflake", account="a", token="t")
    p2 = get_provider("cortex", account="a", token="t")
    assert isinstance(p1, SnowflakeCortexEmbeddings)
    assert isinstance(p2, SnowflakeCortexEmbeddings)
