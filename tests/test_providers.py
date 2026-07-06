"""Tests for embedding providers.

Real-embedding tests run against whatever local endpoint the ``ollama_embed_fn``
fixture (conftest.py) finds — Ollama or an OpenAI-compatible server — and skip
cleanly when neither is available.  Other providers (OpenAI, Cohere, Voyage,
SentenceTransformers) are tested for construction and registry only — they
require API keys or heavy deps.
"""

import numpy as np
import pytest

from zettelkasten_memory.providers import (
    PROVIDER_REGISTRY,
    MalgraEmbeddings,
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
# Real embeddings (local endpoint via conftest fixture)
# ------------------------------------------------------------------


def test_local_embeddings_shape(ollama_embed_fn):
    """The local provider returns correct shape."""
    result = ollama_embed_fn(["hello world", "machine learning"])
    assert result.shape[0] == 2
    assert result.shape[1] > 0
    assert result.dtype == np.float32


def test_local_embeddings_deterministic(ollama_embed_fn):
    """Same input produces the same embedding."""
    r1 = ollama_embed_fn(["test sentence"])
    r2 = ollama_embed_fn(["test sentence"])
    np.testing.assert_array_almost_equal(r1, r2, decimal=5)


def test_local_embeddings_semantic_similarity(ollama_embed_fn):
    """Similar texts produce similar embeddings, dissimilar texts don't."""
    vecs = ollama_embed_fn(
        [
            "the cat sat on the mat",  # 0
            "a kitten rested on the rug",  # 1 — similar to 0
            "quantum physics equations",  # 2 — unrelated
        ]
    )
    # Normalize for cosine similarity
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    vecs = vecs / norms

    sim_related = float(vecs[0] @ vecs[1])
    sim_unrelated = float(vecs[0] @ vecs[2])

    assert sim_related > sim_unrelated, (
        f"Related similarity ({sim_related:.3f}) should exceed "
        f"unrelated similarity ({sim_unrelated:.3f})"
    )


def test_local_embeddings_batching(ollama_embed_fn):
    """Batching splits inputs but returns one array."""
    provider = type(ollama_embed_fn)(
        model=ollama_embed_fn.model, base_url=ollama_embed_fn.base_url, batch_size=2
    )
    texts = ["text one", "text two", "text three", "text four", "text five"]
    result = provider(texts)
    assert result.shape[0] == 5


def test_local_embeddings_single_text(ollama_embed_fn):
    """Single text embedding works."""
    result = ollama_embed_fn(["just one sentence"])
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
    expected = {
        "openai",
        "cohere",
        "voyage",
        "sentence-transformers",
        "local",
        "ollama",
        "snowflake",
        "cortex",
        "malgra",
        "openai-compat",
    }
    assert expected == set(PROVIDER_REGISTRY.keys())


# ------------------------------------------------------------------
# EmbeddingBackend.from_provider()
# ------------------------------------------------------------------


def test_from_provider_ollama():
    """from_provider('ollama') creates a working backend."""
    backend = EmbeddingBackend.from_provider("ollama")
    assert isinstance(backend, EmbeddingBackend)
    assert isinstance(backend._embed_fn, OllamaEmbeddings)


def test_from_provider_ollama_index_and_search(ollama_embed_fn):
    """Full pipeline: build_index → query with real embeddings."""
    backend = EmbeddingBackend(embed_fn=ollama_embed_fn)

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


# ------------------------------------------------------------------
# Malgra / OpenAI-compatible gateway provider
# ------------------------------------------------------------------


def test_malgra_provider_defaults(monkeypatch):
    for var in ("MALGRA_URL", "MALGRA_API_KEY", "MALGRA_AGENT_JWT"):
        monkeypatch.delenv(var, raising=False)
    provider = MalgraEmbeddings()
    assert provider.base_url == "http://127.0.0.1:8766"
    assert provider._api_key == "dummy"  # zero-secret client by design
    assert provider.timeout == 30.0


def test_malgra_provider_env(monkeypatch):
    monkeypatch.setenv("MALGRA_URL", "http://gateway:9000/")
    monkeypatch.setenv("MALGRA_AGENT_JWT", "jwt-token")
    provider = MalgraEmbeddings()
    assert provider.base_url == "http://gateway:9000"
    assert provider._agent_jwt == "jwt-token"


def test_get_provider_malgra_aliases():
    p1 = get_provider("malgra")
    p2 = get_provider("openai-compat")
    assert isinstance(p1, MalgraEmbeddings)
    assert isinstance(p2, MalgraEmbeddings)


def test_malgra_request_shape_and_auth_header():
    """Serve one request with a stub HTTP server; assert path, bearer, ordering."""
    import http.server
    import json
    import threading

    captured: dict = {}

    class Stub(http.server.BaseHTTPRequestHandler):
        def do_POST(self):
            captured["path"] = self.path
            captured["auth"] = self.headers.get("Authorization")
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            captured["body"] = body
            # respond out of order to verify index-based sorting
            data = [
                {"index": 1, "embedding": [0.0, 1.0]},
                {"index": 0, "embedding": [1.0, 0.0]},
            ]
            payload = json.dumps({"data": data}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, *args):
            pass

    server = http.server.HTTPServer(("127.0.0.1", 0), Stub)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        port = server.server_address[1]
        provider = MalgraEmbeddings(
            model="test-model",
            base_url=f"http://127.0.0.1:{port}",
            agent_jwt="agent-jwt-123",
            max_retries=1,
        )
        result = provider(["first", "second"])
    finally:
        server.shutdown()

    assert captured["path"] == "/v1/embeddings"
    assert captured["auth"] == "Bearer agent-jwt-123"  # JWT wins over api_key
    assert captured["body"] == {"model": "test-model", "input": ["first", "second"]}
    np.testing.assert_array_equal(result, np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32))
