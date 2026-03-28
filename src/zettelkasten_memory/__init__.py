"""
Zettelkasten-inspired semantic memory for AI agents.

Works standalone or as a plugin for CrewAI, LangGraph, and Claude Code (MCP).
"""

from .backends import EmbeddingBackend, SearchBackend, TfidfBackend
from .compression import CompressedVectors, TurboQuantCompressor
from .core import SearchResult, Zettel, ZettelMemory
from .providers import (
    CohereEmbeddings,
    OllamaEmbeddings,
    OpenAIEmbeddings,
    SentenceTransformerEmbeddings,
    SnowflakeCortexEmbeddings,
    VoyageEmbeddings,
)

__version__ = "0.1.0"
__all__ = [
    "CohereEmbeddings",
    "CompressedVectors",
    "EmbeddingBackend",
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
