"""
Pluggable search backends for ZettelMemory.

Each backend handles vectorization, similarity search, auto-linking, and tag
extraction.  The core engine delegates these operations through the
``SearchBackend`` protocol so callers can swap TF-IDF for embeddings (or any
future approach) without touching the rest of the system.

Built-in backends:
    - ``TfidfBackend``      – scikit-learn TF-IDF (default, zero-config)
    - ``EmbeddingBackend``   – bring-your-own embedding function
"""

from __future__ import annotations

import base64
from typing import Any, Callable, Protocol, runtime_checkable

import numpy as np

from .compression import CompressedVectors, TurboQuantCompressor


# ------------------------------------------------------------------
# Protocol
# ------------------------------------------------------------------


@runtime_checkable
class SearchBackend(Protocol):
    """Interface that every search backend must satisfy."""

    def build_index(self, ids: list[str], texts: list[str]) -> None:
        """(Re)build the internal index from all current texts."""
        ...

    def query(self, text: str) -> list[tuple[str, float]]:
        """Return ``(zettel_id, similarity)`` pairs for *text*, best first.

        May return an empty list when the index is empty or the query
        produces no signal.
        """
        ...

    def find_similar(self, text: str, threshold: float) -> list[tuple[str, float]]:
        """Return ``(zettel_id, similarity)`` pairs whose similarity >= *threshold*."""
        ...

    def extract_tags(self, text: str) -> set[str]:
        """Derive keyword tags from *text*."""
        ...

    @property
    def needs_rebuild(self) -> bool:
        """True when the index is stale and ``build_index`` should be called."""
        ...

    @needs_rebuild.setter
    def needs_rebuild(self, value: bool) -> None: ...

    def to_dict(self) -> dict[str, Any]:
        """Serialise backend configuration (not the index) for persistence."""
        ...

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SearchBackend":
        """Restore a backend from its serialised config."""
        ...


# ------------------------------------------------------------------
# TF-IDF backend
# ------------------------------------------------------------------

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class TfidfBackend:
    """Search backend powered by scikit-learn TF-IDF vectors."""

    def __init__(
        self,
        max_features: int = 5000,
        ngram_range: tuple[int, int] = (1, 2),
    ) -> None:
        self.max_features = max_features
        self.ngram_range = ngram_range

        self._vectorizer = TfidfVectorizer(
            max_features=max_features,
            stop_words="english",
            ngram_range=ngram_range,
        )
        self._vectors = None
        self._id_order: list[str] = []
        self._dirty = True

    # -- protocol -------------------------------------------------

    @property
    def needs_rebuild(self) -> bool:
        return self._dirty

    @needs_rebuild.setter
    def needs_rebuild(self, value: bool) -> None:
        self._dirty = value

    def build_index(self, ids: list[str], texts: list[str]) -> None:
        if not ids:
            self._vectors = None
            self._id_order = []
            self._dirty = False
            return
        self._id_order = list(ids)
        try:
            self._vectors = self._vectorizer.fit_transform(texts)
            self._dirty = False
        except ValueError:
            self._vectors = None

    def query(self, text: str) -> list[tuple[str, float]]:
        if self._vectors is None or len(self._id_order) == 0:
            return []
        try:
            q_vec = self._vectorizer.transform([text])
            sims = cosine_similarity(q_vec, self._vectors)[0]
        except Exception:
            return []
        pairs = [
            (self._id_order[i], float(sims[i]))
            for i in range(len(self._id_order))
            if sims[i] > 0.0
        ]
        pairs.sort(key=lambda x: x[1], reverse=True)
        return pairs

    def find_similar(self, text: str, threshold: float) -> list[tuple[str, float]]:
        if self._vectors is None or len(self._id_order) == 0:
            return []
        try:
            vec = self._vectorizer.transform([text])
            sims = cosine_similarity(vec, self._vectors)[0]
        except Exception:
            return []
        return [
            (self._id_order[i], float(sims[i]))
            for i in range(len(self._id_order))
            if float(sims[i]) >= threshold
        ]

    def extract_tags(self, text: str) -> set[str]:
        try:
            vec = TfidfVectorizer(max_features=200, stop_words="english")
            tfidf = vec.fit_transform([text])
            names = vec.get_feature_names_out()
            scores = tfidf.toarray()[0]
            top = scores.argsort()[-5:][::-1]
            return {names[i] for i in top if scores[i] > 0}
        except Exception:
            return set()

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "tfidf",
            "max_features": self.max_features,
            "ngram_range": list(self.ngram_range),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TfidfBackend":
        return cls(
            max_features=data.get("max_features", 5000),
            ngram_range=tuple(data.get("ngram_range", [1, 2])),
        )


# ------------------------------------------------------------------
# Embedding backend
# ------------------------------------------------------------------

# Type alias for the embed function users must supply.
# It receives a list of strings and returns an array-like of shape (n, dim).
EmbedFn = Callable[[list[str]], np.ndarray]


class EmbeddingBackend:
    """Search backend powered by a user-supplied embedding function.

    Usage::

        import openai

        client = openai.OpenAI()

        def embed(texts: list[str]) -> np.ndarray:
            resp = client.embeddings.create(input=texts, model="text-embedding-3-small")
            return np.array([e.embedding for e in resp.data])

        mem = ZettelMemory(backend=EmbeddingBackend(embed_fn=embed))

    Any function with signature ``(list[str]) -> np.ndarray`` works — use
    sentence-transformers, Cohere, Voyage, a local ONNX model, etc.

    Optional compression::

        from zettelkasten_memory.compression import TurboQuantCompressor
        backend = EmbeddingBackend(embed_fn=embed, compressor=TurboQuantCompressor())

    Convenience constructor from named provider::

        backend = EmbeddingBackend.from_provider("openai", model="text-embedding-3-small")
    """

    def __init__(
        self,
        embed_fn: EmbedFn | None = None,
        batch_size: int = 64,
        compressor: TurboQuantCompressor | None = None,
    ) -> None:
        self._embed_fn = embed_fn
        self.batch_size = batch_size
        self._compressor = compressor

        self._vectors: np.ndarray | None = None
        self._compressed: CompressedVectors | None = None
        self._id_order: list[str] = []
        self._dirty = True

    # -- convenience constructor ------------------------------------

    @classmethod
    def from_provider(
        cls,
        provider: str,
        *,
        compressor: TurboQuantCompressor | None = None,
        batch_size: int = 64,
        **provider_kwargs: Any,
    ) -> "EmbeddingBackend":
        """Create an EmbeddingBackend from a named provider.

        Example::

            backend = EmbeddingBackend.from_provider("openai", model="text-embedding-3-small")
            backend = EmbeddingBackend.from_provider("sentence-transformers")
        """
        from .providers import get_provider

        embed_fn = get_provider(provider, **provider_kwargs)
        return cls(embed_fn=embed_fn, batch_size=batch_size, compressor=compressor)

    # -- protocol -------------------------------------------------

    @property
    def needs_rebuild(self) -> bool:
        return self._dirty

    @needs_rebuild.setter
    def needs_rebuild(self, value: bool) -> None:
        self._dirty = value

    def build_index(self, ids: list[str], texts: list[str]) -> None:
        if not ids:
            self._vectors = None
            self._compressed = None
            self._id_order = []
            self._dirty = False
            return
        self._id_order = list(ids)

        if self._embed_fn is None:
            raise RuntimeError(
                "EmbeddingBackend.build_index requires embed_fn, but none was "
                "provided. Pass embed_fn when constructing or loading."
            )

        embeddings: list[np.ndarray] = []
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start : start + self.batch_size]
            embeddings.append(np.asarray(self._embed_fn(batch)))
        self._vectors = np.vstack(embeddings)
        # L2-normalise so dot product == cosine similarity
        norms = np.linalg.norm(self._vectors, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        self._vectors = self._vectors / norms

        # Compress if compressor is available
        if self._compressor is not None:
            self._compressed = self._compressor.compress(self._vectors)
        else:
            self._compressed = None

        self._dirty = False

    def query(self, text: str) -> list[tuple[str, float]]:
        if self._id_order and self._vectors is None and self._compressed is not None:
            return self._query_compressed(text)
        if self._vectors is None or len(self._id_order) == 0:
            return []
        if self._embed_fn is None:
            return []
        q_vec = np.asarray(self._embed_fn([text]))
        norm = np.linalg.norm(q_vec)
        if norm > 0:
            q_vec = q_vec / norm

        if self._compressed is not None and self._compressor is not None:
            sims = self._compressor.asymmetric_search(q_vec, self._compressed)
        else:
            sims = (self._vectors @ q_vec.T).flatten()

        pairs = [
            (self._id_order[i], float(sims[i]))
            for i in range(len(self._id_order))
            if sims[i] > 0.0
        ]
        pairs.sort(key=lambda x: x[1], reverse=True)
        return pairs

    def _query_compressed(self, text: str) -> list[tuple[str, float]]:
        """Query using only compressed vectors (no full-precision vectors loaded)."""
        if self._embed_fn is None or self._compressed is None:
            return []
        q_vec = np.asarray(self._embed_fn([text]))
        norm = np.linalg.norm(q_vec)
        if norm > 0:
            q_vec = q_vec / norm

        compressor = self._compressor or TurboQuantCompressor.from_dict({
            "n_bits": 4, "proj_dim": self._compressed.proj_dim, "seed": 42,
        })
        sims = compressor.asymmetric_search(q_vec, self._compressed)
        pairs = [
            (self._id_order[i], float(sims[i]))
            for i in range(len(self._id_order))
            if sims[i] > 0.0
        ]
        pairs.sort(key=lambda x: x[1], reverse=True)
        return pairs

    def find_similar(self, text: str, threshold: float) -> list[tuple[str, float]]:
        if self._vectors is None or len(self._id_order) == 0:
            return []
        if self._embed_fn is None:
            return []
        q_vec = np.asarray(self._embed_fn([text]))
        norm = np.linalg.norm(q_vec)
        if norm > 0:
            q_vec = q_vec / norm

        if self._compressed is not None and self._compressor is not None:
            sims = self._compressor.asymmetric_search(q_vec, self._compressed)
        else:
            sims = (self._vectors @ q_vec.T).flatten()

        return [
            (self._id_order[i], float(sims[i]))
            for i in range(len(self._id_order))
            if float(sims[i]) >= threshold
        ]

    def extract_tags(self, text: str) -> set[str]:
        # Embeddings don't produce term-level importance, so fall back to
        # simple frequency-based extraction (no sklearn needed at runtime).
        try:
            vec = TfidfVectorizer(max_features=200, stop_words="english")
            tfidf = vec.fit_transform([text])
            names = vec.get_feature_names_out()
            scores = tfidf.toarray()[0]
            top = scores.argsort()[-5:][::-1]
            return {names[i] for i in top if scores[i] > 0}
        except Exception:
            return set()

    # -- persistence ------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "type": "embedding",
            "batch_size": self.batch_size,
        }

        # Persist compressor config
        if self._compressor is not None:
            result["compressor"] = self._compressor.to_dict()

        # Persist vectors (compressed or raw float16)
        if self._compressed is not None:
            result["compressed_vectors"] = self._compressed.to_dict()
            result["id_order"] = self._id_order
        elif self._vectors is not None:
            # Store as float16 to save space (2x smaller than float32)
            result["vectors"] = base64.b64encode(
                self._vectors.astype(np.float16).tobytes()
            ).decode()
            result["vectors_shape"] = list(self._vectors.shape)
            result["id_order"] = self._id_order

        return result

    @classmethod
    def from_dict(
        cls,
        data: dict[str, Any],
        embed_fn: EmbedFn | None = None,
    ) -> "EmbeddingBackend":
        # Restore compressor if present
        compressor = None
        if "compressor" in data:
            compressor = TurboQuantCompressor.from_dict(data["compressor"])

        instance = cls(
            embed_fn=embed_fn,
            batch_size=data.get("batch_size", 64),
            compressor=compressor,
        )

        # Restore persisted vectors
        id_order = data.get("id_order", [])
        if "compressed_vectors" in data:
            instance._compressed = CompressedVectors.from_dict(data["compressed_vectors"])
            instance._id_order = id_order
            instance._dirty = False
        elif "vectors" in data:
            shape = tuple(data["vectors_shape"])
            raw = np.frombuffer(
                base64.b64decode(data["vectors"]), dtype=np.float16
            ).reshape(shape)
            instance._vectors = raw.astype(np.float32)
            instance._id_order = id_order
            instance._dirty = False
            # Compress on load if compressor is now provided
            if compressor is not None:
                instance._compressed = compressor.compress(instance._vectors)
        elif id_order:
            # Old format: had ids but no vectors — need rebuild
            instance._id_order = id_order
            instance._dirty = True
        else:
            # No vectors at all — mark dirty so build_index runs on first search
            instance._dirty = True

        return instance


# ------------------------------------------------------------------
# Registry — resolve backend type string to class
# ------------------------------------------------------------------

BACKEND_REGISTRY: dict[str, type] = {
    "tfidf": TfidfBackend,
    "embedding": EmbeddingBackend,
}


def backend_from_dict(data: dict[str, Any], **kwargs: Any) -> SearchBackend:
    """Reconstruct a backend from its serialised config dict."""
    btype = data.get("type", "tfidf")
    cls = BACKEND_REGISTRY.get(btype)
    if cls is None:
        raise ValueError(f"Unknown backend type: {btype!r}")
    if btype == "embedding":
        return cls.from_dict(data, embed_fn=kwargs.get("embed_fn"))  # type: ignore[arg-type]
    return cls.from_dict(data)
