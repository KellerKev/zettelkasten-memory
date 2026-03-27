"""
Zettelkasten-inspired semantic memory for AI agents.

Works standalone or as a plugin for CrewAI, LangGraph, and Claude Code (MCP).
"""

from .backends import EmbeddingBackend, SearchBackend, TfidfBackend
from .core import SearchResult, Zettel, ZettelMemory

__version__ = "0.1.0"
__all__ = [
    "EmbeddingBackend",
    "SearchBackend",
    "SearchResult",
    "TfidfBackend",
    "Zettel",
    "ZettelMemory",
]
