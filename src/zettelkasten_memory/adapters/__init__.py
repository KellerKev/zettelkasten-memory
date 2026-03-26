"""
Adapters for integrating ZettelMemory with popular agent frameworks.

Each adapter is lazily imported to avoid hard dependencies:

    from zettelkasten_memory.adapters.crewai import ZettelStorage
    from zettelkasten_memory.adapters.langgraph import ZettelStore
    from zettelkasten_memory.adapters.mcp_server import create_mcp_server
"""
