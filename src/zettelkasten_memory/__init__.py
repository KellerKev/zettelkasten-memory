"""
Zettelkasten-inspired semantic memory for AI agents.

Works standalone or as a plugin for CrewAI, LangGraph, and Claude Code (MCP).
"""

from .backends import EmbeddingBackend, SearchBackend, TfidfBackend
from .camouflage import CamouflageCodec, CamouflageError
from .compression import CompressedVectors, TurboQuantCompressor
from .core import SearchResult, Zettel, ZettelMemory
from .crypto import EncryptionError, KeyNotFoundError
from .providers import (
    CohereEmbeddings,
    MalgraEmbeddings,
    OllamaEmbeddings,
    OpenAIEmbeddings,
    SentenceTransformerEmbeddings,
    SnowflakeCortexEmbeddings,
    VoyageEmbeddings,
)

__version__ = "0.2.0"
__all__ = [
    "CamouflageCodec",
    "CamouflageError",
    "CohereEmbeddings",
    "CompressedVectors",
    "EmbeddingBackend",
    "EncryptionError",
    "KeyNotFoundError",
    "MalgraEmbeddings",
    "OllamaEmbeddings",
    "OpenAIEmbeddings",
    "SearchBackend",
    "SearchResult",
    "SentenceTransformerEmbeddings",
    "SnowflakeCortexEmbeddings",
    "TfidfBackend",
    "TurboQuantCompressor",
    "VoyageEmbeddings",
    "Zettel",
    "ZettelMemory",
]
