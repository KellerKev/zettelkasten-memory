# zettelkasten-memory

Zettelkasten-inspired semantic memory for AI agents. Works standalone or as a plugin for **CrewAI**, **LangGraph**, and **Claude Code** (via MCP).

Each memory is a "zettel" (note) that is automatically tagged, scored by importance, and **linked to related memories** using TF-IDF cosine similarity. Search results are ranked by a combination of text similarity, importance, recency, and graph connectivity — so well-connected memories surface more readily.

## Install

```bash
# Core only
pip install -e .

# Or with pixi
pixi install

# With framework adapters
pixi install -e crewai
pixi install -e langgraph
pixi install -e mcp
```

## Standalone Usage

```python
from zettelkasten_memory import ZettelMemory

mem = ZettelMemory()

# Store memories — they auto-link when semantically similar
mem.add("The project uses FastAPI for the REST API layer")
mem.add("JWT authentication is used on all API endpoints")
mem.add("The user prefers concise answers")

# Search — returns results ranked by similarity + importance + connections
results = mem.search("what framework does the project use?")
for r in results:
    print(f"{r.score:.3f}  {r.zettel.content}")

# Get formatted context string for an LLM prompt
context = mem.get_context("how does auth work?", max_tokens=2000)

# Traverse the link graph
connected = mem.get_connected(results[0].zettel.id, depth=2)

# Persistence
mem.save("memory.json")
mem = ZettelMemory.load("memory.json")
```

## CrewAI Adapter

Drop-in replacement for CrewAI's memory storage:

```python
from crewai import Crew, Agent, Task
from crewai.memory import ShortTermMemory
from zettelkasten_memory.adapters.crewai import ZettelStorage

storage = ZettelStorage(persist_path="crew_memory.json")

crew = Crew(
    agents=[...],
    tasks=[...],
    memory=True,
    short_term_memory=ShortTermMemory(storage=storage),
)
```

The `ZettelStorage` class implements `save()`, `search()`, and `reset()` — the interface CrewAI expects from a storage backend. Memories auto-link across tasks, so agents benefit from connections discovered in earlier steps.

## LangGraph Adapter

Implements LangGraph's `BaseStore` interface for the Store API:

```python
from langgraph.graph import StateGraph
from zettelkasten_memory.adapters.langgraph import ZettelStore

store = ZettelStore(persist_path="langgraph_memory.json")
app = graph.compile(store=store)

# In your node functions:
async def my_node(state, *, store):
    results = await store.asearch(("memories", "user_123"), query="project architecture")
    await store.aput(("memories", "user_123"), "key1", {"content": "Uses FastAPI"})
```

Namespaces are mapped to zettel metadata so memories from different users/threads stay isolated.

## Claude Code / MCP Server

Run as an MCP server that gives Claude persistent, searchable memory:

```bash
# Start the server
pixi run -e mcp python -m zettelkasten_memory.adapters.mcp_server --persist memory.json
```

Configure in `~/.claude/settings.json`:

```json
{
  "mcpServers": {
    "zettel-memory": {
      "command": "pixi",
      "args": ["run", "-e", "mcp", "python", "-m",
               "zettelkasten_memory.adapters.mcp_server",
               "--persist", "/path/to/memory.json"]
    }
  }
}
```

This exposes six tools to Claude:

| Tool | Description |
|---|---|
| `memory_store` | Save a new memory with optional tags and importance |
| `memory_search` | Search by semantic similarity |
| `memory_get` | Retrieve a specific memory by ID |
| `memory_delete` | Delete a memory and clean up links |
| `memory_connections` | Traverse the link graph (N hops) |
| `memory_stats` | Get memory statistics |

## How It Works

**Storage:** Each zettel has content, metadata, tags, importance (0-1), and a set of bidirectional connections to other zettels.

**Auto-linking:** When a new zettel is added, its TF-IDF vector is compared against all existing zettels. Any pair exceeding the `connection_threshold` (default 0.25 cosine similarity) gets a bidirectional link.

**Search ranking:** `score = tfidf_similarity * importance * recency_factor * connection_boost`

- `recency_factor` decays over days since last access
- `connection_boost` rewards well-connected zettels (up to 2x for 10+ links)
- Falls back to keyword overlap when TF-IDF yields no matches

**Eviction:** When over capacity, the least-valuable zettels are removed. Value = `importance * (1 + access_count) * recency`.

**Persistence:** JSON file with all zettels and config. Load/save are explicit — no background I/O.

## Configuration

```python
mem = ZettelMemory(
    max_zettels=5000,          # capacity before eviction kicks in
    connection_threshold=0.25,  # min cosine similarity to auto-link
)
```

## Development

```bash
pixi install -e dev
pixi run test      # pytest
pixi run format    # black
```

## License

MIT
