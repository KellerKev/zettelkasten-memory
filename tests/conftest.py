"""Shared fixtures for all tests.

Embedding-integration tests need a real local embedding endpoint.  The
``embed_fn`` fixture probes, in order:

1. Ollama (``http://localhost:11434``, model ``nomic-embed-text``)
2. Any OpenAI-compatible server such as llama.cpp
   (``ZETTEL_TEST_EMBED_URL``, default ``http://localhost:8092``;
   model from ``ZETTEL_TEST_EMBED_MODEL``)

If neither responds, embedding tests are skipped instead of failing, so the
TF-IDF-only suite stays green on machines without a local model server.
"""

import os

import pytest

from zettelkasten_memory.backends import EmbeddingBackend
from zettelkasten_memory.providers import MalgraEmbeddings, OllamaEmbeddings


def _probe(provider) -> bool:
    try:
        provider(["probe"])
        return True
    except Exception:
        return False


@pytest.fixture(scope="session")
def ollama_embed_fn():
    """Real embedding function using a local model server (Ollama or llama.cpp)."""
    ollama = OllamaEmbeddings(model="nomic-embed-text", max_retries=1, timeout=5.0)
    if _probe(ollama):
        return ollama

    llamacpp = MalgraEmbeddings(
        model=os.environ.get("ZETTEL_TEST_EMBED_MODEL", "nomic-embed-text"),
        base_url=os.environ.get("ZETTEL_TEST_EMBED_URL", "http://localhost:8092"),
        max_retries=1,
        timeout=5.0,
    )
    if _probe(llamacpp):
        return llamacpp

    pytest.skip(
        "no local embedding endpoint: need Ollama on :11434 or an "
        "OpenAI-compatible server on ZETTEL_TEST_EMBED_URL (default :8092)"
    )


@pytest.fixture(scope="session")
def ollama_dim(ollama_embed_fn):
    """Embedding dimension from the local model."""
    result = ollama_embed_fn(["dim probe"])
    return result.shape[1]


@pytest.fixture
def embedding_backend(ollama_embed_fn):
    """Fresh EmbeddingBackend wired to the local embedding endpoint."""
    return EmbeddingBackend(embed_fn=ollama_embed_fn)
