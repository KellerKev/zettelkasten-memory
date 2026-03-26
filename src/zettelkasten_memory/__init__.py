"""
Zettelkasten-inspired semantic memory for AI agents.

Works standalone or as a plugin for CrewAI, LangGraph, and Claude Code (MCP).
"""

from .core import ZettelMemory, Zettel, SearchResult

__version__ = "0.1.0"
__all__ = ["ZettelMemory", "Zettel", "SearchResult"]
