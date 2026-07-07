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
            (self._id_order[i], float(sims[i])) for i in range(len(self._id_order)) if sims[i] > 0.0
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
            (self._id_order[i], float(sims[i])) for i in range(len(self._id_order)) if sims[i] > 0.0
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

        compressor = self._compressor or TurboQuantCompressor.from_dict(
            {
                "n_bits": 4,
                "proj_dim": self._compressed.proj_dim,
                "seed": 42,
            }
        )
        sims = compressor.asymmetric_search(q_vec, self._compressed)
        pairs = [
            (self._id_order[i], float(sims[i])) for i in range(len(self._id_order)) if sims[i] > 0.0
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
            raw = np.frombuffer(base64.b64decode(data["vectors"]), dtype=np.float16).reshape(shape)
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
# Hybrid backend
# ------------------------------------------------------------------


class HybridBackend:
    """Combine TF-IDF keyword search with embedding semantic search.

    Each query runs through both sub-backends and their result lists are merged
    with **reciprocal rank fusion** (RRF): an id's fused score is
    ``sum_b weight_b / (rrf_k + rank_b)`` over the backends that ranked it. RRF
    is scale-free, so it sidesteps the fact that TF-IDF cosine and embedding
    dot products live on different scales — no per-backend normalisation, and a
    document that ranks well in *either* modality surfaces. This also removes
    the old all-or-nothing keyword fallback: exact-term hits (TF-IDF) and
    paraphrase hits (embeddings) contribute to one ranking.

    Auto-linking (``find_similar``) instead delegates to the semantic backend,
    because ``connection_threshold`` is calibrated against cosine similarity
    (0..1), not fused RRF scores.

    Usage::

        backend = HybridBackend(embed_fn=my_embed_fn)
        backend = HybridBackend(embedding=EmbeddingBackend.from_provider("ollama"))
        mem = ZettelMemory(backend=backend)
    """

    def __init__(
        self,
        embed_fn: EmbedFn | None = None,
        *,
        tfidf: "TfidfBackend | None" = None,
        embedding: "EmbeddingBackend | None" = None,
        tfidf_weight: float = 1.0,
        embedding_weight: float = 1.0,
        rrf_k: int = 60,
    ) -> None:
        if embedding is None and embed_fn is None:
            raise ValueError(
                "HybridBackend needs an embedding source: pass embed_fn= or embedding="
            )
        self._tfidf = tfidf or TfidfBackend()
        self._embedding = embedding or EmbeddingBackend(embed_fn=embed_fn)
        self.tfidf_weight = float(tfidf_weight)
        self.embedding_weight = float(embedding_weight)
        self.rrf_k = int(rrf_k)

    # -- protocol -------------------------------------------------

    @property
    def needs_rebuild(self) -> bool:
        return self._tfidf.needs_rebuild or self._embedding.needs_rebuild

    @needs_rebuild.setter
    def needs_rebuild(self, value: bool) -> None:
        self._tfidf.needs_rebuild = value
        self._embedding.needs_rebuild = value

    def build_index(self, ids: list[str], texts: list[str]) -> None:
        self._tfidf.build_index(ids, texts)
        self._embedding.build_index(ids, texts)

    def _rrf(self, pairs: list[tuple[str, float]], weight: float) -> dict[str, float]:
        """Reciprocal-rank contribution of one backend's ranked results."""
        out: dict[str, float] = {}
        for rank, (zid, _score) in enumerate(pairs):
            out[zid] = weight / (self.rrf_k + rank)
        return out

    def query(self, text: str) -> list[tuple[str, float]]:
        t = self._rrf(self._tfidf.query(text), self.tfidf_weight)
        e = self._rrf(self._embedding.query(text), self.embedding_weight)
        fused: dict[str, float] = dict(t)
        for zid, contrib in e.items():
            fused[zid] = fused.get(zid, 0.0) + contrib
        pairs = sorted(fused.items(), key=lambda kv: kv[1], reverse=True)
        return pairs

    def find_similar(self, text: str, threshold: float) -> list[tuple[str, float]]:
        # Linking thresholds are calibrated on cosine similarity; use the
        # semantic backend, falling back to TF-IDF only if it has no vectors.
        similar = self._embedding.find_similar(text, threshold)
        if similar:
            return similar
        return self._tfidf.find_similar(text, threshold)

    def extract_tags(self, text: str) -> set[str]:
        return self._tfidf.extract_tags(text)

    # -- persistence ------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "hybrid",
            "tfidf": self._tfidf.to_dict(),
            "embedding": self._embedding.to_dict(),
            "tfidf_weight": self.tfidf_weight,
            "embedding_weight": self.embedding_weight,
            "rrf_k": self.rrf_k,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any], embed_fn: EmbedFn | None = None) -> "HybridBackend":
        tfidf = TfidfBackend.from_dict(data.get("tfidf", {"type": "tfidf"}))
        embedding = EmbeddingBackend.from_dict(
            data.get("embedding", {"type": "embedding"}), embed_fn=embed_fn
        )
        return cls(
            tfidf=tfidf,
            embedding=embedding,
            tfidf_weight=data.get("tfidf_weight", 1.0),
            embedding_weight=data.get("embedding_weight", 1.0),
            rrf_k=data.get("rrf_k", 60),
        )


# ------------------------------------------------------------------
# FAISS backend
# ------------------------------------------------------------------


def _require_faiss():
    try:
        import faiss

        return faiss
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "the 'faiss-cpu' package is required for FaissBackend. "
            "Install with: pip install 'zettelkasten-memory[faiss]'"
        ) from exc


class FaissBackend:
    """Semantic search backed by a FAISS index — scales past the brute-force
    in-memory backend.

    Cosine similarity via inner product on L2-normalised embeddings. Two index
    types:

    - ``"flat"`` (default): exact inner-product search — same results as
      ``EmbeddingBackend`` but through FAISS's optimised, compact index.
    - ``"hnsw"``: approximate nearest-neighbour graph that scales to very large
      stores; each query returns the top ``search_k`` candidates (the composite
      score then reranks those).

    The built index serialises with the store, so a reload restores it without
    re-embedding (``embed_fn`` is only needed to add/search new text).

    Requires ``faiss-cpu`` (extra: ``zettelkasten-memory[faiss]``).
    """

    def __init__(
        self,
        embed_fn: EmbedFn | None = None,
        *,
        batch_size: int = 64,
        index: str = "flat",
        hnsw_m: int = 32,
        search_k: int = 64,
    ) -> None:
        _require_faiss()
        if index not in ("flat", "hnsw"):
            raise ValueError("index must be 'flat' or 'hnsw'")
        self._embed_fn = embed_fn
        self.batch_size = batch_size
        self.index_type = index
        self.hnsw_m = int(hnsw_m)
        self.search_k = int(search_k)
        self._index = None
        self._id_order: list[str] = []
        self._dim: int | None = None
        self._dirty = True

    @classmethod
    def from_provider(cls, provider: str, **kwargs: Any) -> "FaissBackend":
        from .providers import get_provider

        index = kwargs.pop("index", "flat")
        return cls(embed_fn=get_provider(provider, **kwargs), index=index)

    # -- protocol -------------------------------------------------

    @property
    def needs_rebuild(self) -> bool:
        return self._dirty

    @needs_rebuild.setter
    def needs_rebuild(self, value: bool) -> None:
        self._dirty = value

    def _new_index(self, dim: int):
        faiss = _require_faiss()
        if self.index_type == "hnsw":
            return faiss.IndexHNSWFlat(dim, self.hnsw_m, faiss.METRIC_INNER_PRODUCT)
        return faiss.IndexFlatIP(dim)

    def _embed_norm(self, texts: list[str]) -> np.ndarray:
        vecs = np.asarray(self._embed_fn(texts), dtype=np.float32)  # type: ignore[misc]
        if vecs.ndim == 1:
            vecs = vecs.reshape(1, -1)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return (vecs / norms).astype(np.float32)

    def build_index(self, ids: list[str], texts: list[str]) -> None:
        if not ids:
            self._index = None
            self._id_order = []
            self._dirty = False
            return
        if self._embed_fn is None:
            # can't (re)embed without a function; keep any loaded index as-is
            self._dirty = False
            return
        vecs = self._embed_norm(texts)
        self._dim = int(vecs.shape[1])
        index = self._new_index(self._dim)
        index.add(vecs)
        self._index = index
        self._id_order = list(ids)
        self._dirty = False

    def _search(self, text: str, k: int) -> list[tuple[str, float]]:
        if self._index is None or not self._id_order or self._embed_fn is None:
            return []
        q = self._embed_norm([text])
        k = max(1, min(k, len(self._id_order)))
        scores, idxs = self._index.search(q, k)
        out: list[tuple[str, float]] = []
        for score, i in zip(scores[0], idxs[0]):
            if i == -1:
                continue
            out.append((self._id_order[int(i)], float(score)))
        return out

    def query(self, text: str) -> list[tuple[str, float]]:
        # flat is exact, so return every positive match; hnsw returns top-k
        k = len(self._id_order) if self.index_type == "flat" else self.search_k
        return [(zid, s) for zid, s in self._search(text, k) if s > 0.0]

    def find_similar(self, text: str, threshold: float) -> list[tuple[str, float]]:
        k = len(self._id_order) if self.index_type == "flat" else self.search_k
        return [(zid, s) for zid, s in self._search(text, k) if s >= threshold]

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

    # -- persistence ------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "type": "faiss",
            "batch_size": self.batch_size,
            "index_type": self.index_type,
            "hnsw_m": self.hnsw_m,
            "search_k": self.search_k,
        }
        if self._index is not None and self._id_order:
            faiss = _require_faiss()
            blob = faiss.serialize_index(self._index)
            result["index"] = base64.b64encode(bytes(blob)).decode()
            result["id_order"] = self._id_order
            result["dim"] = self._dim
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any], embed_fn: EmbedFn | None = None) -> "FaissBackend":
        instance = cls(
            embed_fn=embed_fn,
            batch_size=data.get("batch_size", 64),
            index=data.get("index_type", "flat"),
            hnsw_m=data.get("hnsw_m", 32),
            search_k=data.get("search_k", 64),
        )
        if "index" in data and data.get("id_order"):
            faiss = _require_faiss()
            raw = np.frombuffer(base64.b64decode(data["index"]), dtype=np.uint8)
            instance._index = faiss.deserialize_index(raw)
            instance._id_order = data["id_order"]
            instance._dim = data.get("dim")
            instance._dirty = False
        return instance


# ------------------------------------------------------------------
# Registry — resolve backend type string to class
# ------------------------------------------------------------------

BACKEND_REGISTRY: dict[str, type] = {
    "tfidf": TfidfBackend,
    "embedding": EmbeddingBackend,
    "hybrid": HybridBackend,
    "faiss": FaissBackend,
}


def backend_from_dict(data: dict[str, Any], **kwargs: Any) -> SearchBackend:
    """Reconstruct a backend from its serialised config dict."""
    btype = data.get("type", "tfidf")
    cls = BACKEND_REGISTRY.get(btype)
    if cls is None:
        raise ValueError(f"Unknown backend type: {btype!r}")
    if btype in ("embedding", "hybrid", "faiss"):
        return cls.from_dict(data, embed_fn=kwargs.get("embed_fn"))  # type: ignore[arg-type]
    return cls.from_dict(data)
