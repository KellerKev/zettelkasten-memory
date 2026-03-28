"""Shared fixtures for all tests.

Uses Ollama with nomic-embed-text for real embeddings.
"""

import pytest
import numpy as np

from zettelkasten_memory.providers import OllamaEmbeddings
from zettelkasten_memory.backends import EmbeddingBackend


@pytest.fixture(scope="session")
def ollama_embed_fn():
    """Real embedding function using local Ollama."""
    return OllamaEmbeddings(model="nomic-embed-text")


@pytest.fixture(scope="session")
def ollama_dim(ollama_embed_fn):
    """Embedding dimension from the Ollama model."""
    result = ollama_embed_fn(["dim probe"])
    return result.shape[1]


@pytest.fixture
def embedding_backend(ollama_embed_fn):
    """Fresh EmbeddingBackend wired to Ollama."""
    return EmbeddingBackend(embed_fn=ollama_embed_fn)
