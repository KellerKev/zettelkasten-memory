"""
Pre-built embedding providers for EmbeddingBackend.

Each provider implements ``__call__(texts: list[str]) -> np.ndarray`` so it
can be passed directly as ``embed_fn``::

    from zettelkasten_memory import EmbeddingBackend
    from zettelkasten_memory.providers import OpenAIEmbeddings

    backend = EmbeddingBackend(embed_fn=OpenAIEmbeddings())
    # or via convenience constructor:
    backend = EmbeddingBackend.from_provider("openai")

All providers read API keys from environment variables by default, but also
accept them as constructor parameters.  External dependencies are imported
lazily — a missing package gives a clear error message.
"""

from __future__ import annotations

import os
import time
from typing import Any

import numpy as np


# ------------------------------------------------------------------
# Base
# ------------------------------------------------------------------


class _BaseProvider:
    """Shared retry / batching logic."""

    def __init__(self, *, batch_size: int = 64, max_retries: int = 3) -> None:
        self.batch_size = batch_size
        self.max_retries = max_retries

    def __call__(self, texts: list[str]) -> np.ndarray:
        all_embeddings: list[np.ndarray] = []
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start : start + self.batch_size]
            emb = self._embed_with_retry(batch)
            all_embeddings.append(np.asarray(emb, dtype=np.float32))
        return np.vstack(all_embeddings)

    def _embed_with_retry(self, texts: list[str]) -> np.ndarray:
        last_err: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                return self._embed_batch(texts)
            except Exception as exc:
                last_err = exc
                if attempt < self.max_retries - 1:
                    time.sleep(2**attempt * 0.5)
        raise RuntimeError(
            f"{self.__class__.__name__}: all {self.max_retries} retries failed"
        ) from last_err

    def _embed_batch(self, texts: list[str]) -> np.ndarray:
        raise NotImplementedError


# ------------------------------------------------------------------
# OpenAI
# ------------------------------------------------------------------


class OpenAIEmbeddings(_BaseProvider):
    """OpenAI embeddings provider.

    Requires ``openai`` package and ``OPENAI_API_KEY`` env var (or pass *api_key*).

    Usage::

        embed_fn = OpenAIEmbeddings()                          # defaults
        embed_fn = OpenAIEmbeddings(model="text-embedding-3-large")
    """

    def __init__(
        self,
        model: str = "text-embedding-3-small",
        *,
        api_key: str | None = None,
        batch_size: int = 64,
        max_retries: int = 3,
    ) -> None:
        super().__init__(batch_size=batch_size, max_retries=max_retries)
        self.model = model
        self._api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self._client = None

    def _get_client(self):
        if self._client is None:
            try:
                from openai import OpenAI
            except ImportError:
                raise ImportError(
                    "OpenAIEmbeddings requires the 'openai' package. "
                    "Install it with: pip install openai"
                )
            self._client = OpenAI(api_key=self._api_key)
        return self._client

    def _embed_batch(self, texts: list[str]) -> np.ndarray:
        client = self._get_client()
        resp = client.embeddings.create(input=texts, model=self.model)
        return np.array([e.embedding for e in resp.data], dtype=np.float32)


# ------------------------------------------------------------------
# Cohere
# ------------------------------------------------------------------


class CohereEmbeddings(_BaseProvider):
    """Cohere embeddings provider.

    Requires ``cohere`` package and ``COHERE_API_KEY`` env var (or pass *api_key*).

    Usage::

        embed_fn = CohereEmbeddings()
        embed_fn = CohereEmbeddings(model="embed-multilingual-v3.0")
    """

    def __init__(
        self,
        model: str = "embed-english-v3.0",
        *,
        input_type: str = "search_document",
        api_key: str | None = None,
        batch_size: int = 64,
        max_retries: int = 3,
    ) -> None:
        super().__init__(batch_size=batch_size, max_retries=max_retries)
        self.model = model
        self.input_type = input_type
        self._api_key = api_key or os.environ.get("COHERE_API_KEY")
        self._client = None

    def _get_client(self):
        if self._client is None:
            try:
                import cohere
            except ImportError:
                raise ImportError(
                    "CohereEmbeddings requires the 'cohere' package. "
                    "Install it with: pip install cohere"
                )
            self._client = cohere.Client(api_key=self._api_key)
        return self._client

    def _embed_batch(self, texts: list[str]) -> np.ndarray:
        client = self._get_client()
        resp = client.embed(
            texts=texts,
            model=self.model,
            input_type=self.input_type,
        )
        return np.array(resp.embeddings, dtype=np.float32)


# ------------------------------------------------------------------
# Voyage
# ------------------------------------------------------------------


class VoyageEmbeddings(_BaseProvider):
    """Voyage AI embeddings provider.

    Requires ``voyageai`` package and ``VOYAGE_API_KEY`` env var (or pass *api_key*).

    Usage::

        embed_fn = VoyageEmbeddings()
        embed_fn = VoyageEmbeddings(model="voyage-3-lite")
    """

    def __init__(
        self,
        model: str = "voyage-3",
        *,
        api_key: str | None = None,
        batch_size: int = 64,
        max_retries: int = 3,
    ) -> None:
        super().__init__(batch_size=batch_size, max_retries=max_retries)
        self.model = model
        self._api_key = api_key or os.environ.get("VOYAGE_API_KEY")
        self._client = None

    def _get_client(self):
        if self._client is None:
            try:
                import voyageai
            except ImportError:
                raise ImportError(
                    "VoyageEmbeddings requires the 'voyageai' package. "
                    "Install it with: pip install voyageai"
                )
            self._client = voyageai.Client(api_key=self._api_key)
        return self._client

    def _embed_batch(self, texts: list[str]) -> np.ndarray:
        client = self._get_client()
        resp = client.embed(texts, model=self.model)
        return np.array(resp.embeddings, dtype=np.float32)


# ------------------------------------------------------------------
# Sentence Transformers (local)
# ------------------------------------------------------------------


class SentenceTransformerEmbeddings(_BaseProvider):
    """Local sentence-transformers provider.

    Requires ``sentence-transformers`` package.  No API key needed.

    Usage::

        embed_fn = SentenceTransformerEmbeddings()
        embed_fn = SentenceTransformerEmbeddings(model="all-mpnet-base-v2")
    """

    def __init__(
        self,
        model: str = "all-MiniLM-L6-v2",
        *,
        batch_size: int = 64,
        max_retries: int = 3,
    ) -> None:
        super().__init__(batch_size=batch_size, max_retries=max_retries)
        self.model_name = model
        self._model = None

    def _get_model(self):
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError:
                raise ImportError(
                    "SentenceTransformerEmbeddings requires the 'sentence-transformers' "
                    "package. Install it with: pip install sentence-transformers"
                )
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def _embed_batch(self, texts: list[str]) -> np.ndarray:
        model = self._get_model()
        return np.asarray(model.encode(texts), dtype=np.float32)


# ------------------------------------------------------------------
# Ollama (local)
# ------------------------------------------------------------------


class OllamaEmbeddings(_BaseProvider):
    """Ollama local embeddings provider.

    Calls the Ollama REST API at ``base_url`` (default ``http://localhost:11434``).
    No API key needed — just a running Ollama instance with an embedding model.

    Usage::

        embed_fn = OllamaEmbeddings()                            # defaults to nomic-embed-text
        embed_fn = OllamaEmbeddings(model="mxbai-embed-large")
    """

    def __init__(
        self,
        model: str = "nomic-embed-text",
        *,
        base_url: str = "http://localhost:11434",
        batch_size: int = 64,
        max_retries: int = 3,
    ) -> None:
        super().__init__(batch_size=batch_size, max_retries=max_retries)
        self.model = model
        self.base_url = base_url.rstrip("/")

    def _embed_batch(self, texts: list[str]) -> np.ndarray:
        import urllib.request
        import json as _json

        payload = _json.dumps({"model": self.model, "input": texts}).encode()
        req = urllib.request.Request(
            f"{self.base_url}/api/embed",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req) as resp:
            data = _json.loads(resp.read())
        return np.array(data["embeddings"], dtype=np.float32)


# ------------------------------------------------------------------
# Snowflake Cortex
# ------------------------------------------------------------------


class SnowflakeCortexEmbeddings(_BaseProvider):
    """Snowflake Cortex embedding provider.

    Calls the Cortex REST API using a PAT (Programmatic Access Token).
    No extra packages needed — uses ``urllib`` directly.

    Usage::

        embed_fn = SnowflakeCortexEmbeddings(
            account="myorg-myaccount",   # or set SNOWFLAKE_ACCOUNT
            token="pat-secret-here",     # or set SNOWFLAKE_PAT_TOKEN
        )
        embed_fn = SnowflakeCortexEmbeddings(model="snowflake-arctic-embed-l-v2.0")

    Available models:
        - ``snowflake-arctic-embed-m-v1.5`` (768 dims, default)
        - ``snowflake-arctic-embed-m`` (768 dims)
        - ``snowflake-arctic-embed-l-v2.0`` (1024 dims, multilingual)
        - ``e5-base-v2`` (768 dims)
        - ``voyage-multilingual-2`` (1024 dims, 32k context)
        - ``nv-embed-qa-4`` (1024 dims)
    """

    def __init__(
        self,
        model: str = "snowflake-arctic-embed-m-v1.5",
        *,
        account: str | None = None,
        token: str | None = None,
        batch_size: int = 512,
        max_retries: int = 3,
    ) -> None:
        super().__init__(batch_size=batch_size, max_retries=max_retries)
        self.model = model
        self._account = account or os.environ.get("SNOWFLAKE_ACCOUNT")
        self._token = token or os.environ.get("SNOWFLAKE_PAT_TOKEN")

    def _embed_batch(self, texts: list[str]) -> np.ndarray:
        import urllib.request
        import json as _json

        if not self._account:
            raise ValueError(
                "SnowflakeCortexEmbeddings requires 'account' parameter or "
                "SNOWFLAKE_ACCOUNT env var (format: orgname-accountname)"
            )
        if not self._token:
            raise ValueError(
                "SnowflakeCortexEmbeddings requires 'token' parameter or "
                "SNOWFLAKE_PAT_TOKEN env var"
            )

        url = (
            f"https://{self._account}.snowflakecomputing.com"
            f"/api/v2/cortex/inference:embed"
        )
        payload = _json.dumps({"text": texts, "model": self.model}).encode()
        req = urllib.request.Request(
            url,
            data=payload,
            headers={
                "Content-Type": "application/json",
                "Accept": "application/json",
                "Authorization": f"Bearer {self._token}",
                "X-Snowflake-Authorization-Token-Type": "PROGRAMMATIC_ACCESS_TOKEN",
            },
        )
        with urllib.request.urlopen(req) as resp:
            data = _json.loads(resp.read())

        # Response: {"data": [{"embedding": [[...]], "index": 0}, ...]}
        # Sort by index to maintain input order, flatten nested embedding array
        items = sorted(data["data"], key=lambda x: x["index"])
        embeddings = [item["embedding"][0] for item in items]
        return np.array(embeddings, dtype=np.float32)


# ------------------------------------------------------------------
# Registry for from_provider()
# ------------------------------------------------------------------

PROVIDER_REGISTRY: dict[str, type[_BaseProvider]] = {
    "openai": OpenAIEmbeddings,
    "cohere": CohereEmbeddings,
    "voyage": VoyageEmbeddings,
    "sentence-transformers": SentenceTransformerEmbeddings,
    "local": SentenceTransformerEmbeddings,
    "ollama": OllamaEmbeddings,
    "snowflake": SnowflakeCortexEmbeddings,
    "cortex": SnowflakeCortexEmbeddings,
}


def get_provider(name: str, **kwargs: Any) -> _BaseProvider:
    """Look up a provider by name and instantiate it with *kwargs*."""
    cls = PROVIDER_REGISTRY.get(name.lower())
    if cls is None:
        available = ", ".join(sorted(PROVIDER_REGISTRY.keys()))
        raise ValueError(
            f"Unknown provider {name!r}. Available: {available}"
        )
    return cls(**kwargs)
