# zettelkasten-memory

Zettelkasten-inspired semantic memory for AI agents. Works standalone or as a drop-in plugin for **CrewAI**, **LangGraph**, and **Claude Code** (via MCP).

Each memory is a "zettel" (note) that is automatically tagged, scored by importance, and **linked to related memories** by semantic similarity. Search results are ranked by a composite score of text similarity, importance, recency, and graph connectivity — so well-connected, important memories surface first.

Supports **pluggable search backends**: use the built-in TF-IDF backend (zero config) or bring your own embedding function (OpenAI, Cohere, Voyage, sentence-transformers, local ONNX — anything goes).

---

## Table of Contents

- [Install](#install)
- [Quick Start](#quick-start)
- [Embedding Backend](#embedding-backend)
- [CrewAI Adapter](#crewai-adapter)
- [LangGraph Adapter](#langgraph-adapter)
- [Claude Code / MCP Server](#claude-code--mcp-server)
- [How It Works](#how-it-works)
- [Configuration](#configuration)
- [Development](#development)
- [Roadmap](#roadmap)
- [License](#license)

---

## Install

```bash
# Core (TF-IDF backend, no API keys needed)
pip install -e .

# Or with pixi (recommended)
pixi install

# With framework adapters
pixi install -e crewai       # CrewAI integration
pixi install -e langgraph    # LangGraph integration
pixi install -e mcp          # MCP server for Claude Code
pixi install -e embeddings   # OpenAI embeddings support
pixi install -e dev          # Development (pytest, black)
```

---

## Quick Start

```python
from zettelkasten_memory import ZettelMemory

mem = ZettelMemory()

# Store memories — they auto-link when semantically similar
mem.add("The project uses FastAPI for the REST API layer")
mem.add("JWT authentication is used on all API endpoints")
mem.add("The user prefers concise answers")

# Search — ranked by similarity + importance + recency + connections
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

### Adding memories with metadata

```python
z = mem.add(
    "Deploy to production requires two approvals",
    tags={"process", "deploy"},
    importance=0.9,                          # 0.0 - 1.0
    metadata={"source": "team-handbook"},
)
print(z.id, z.tags, z.connections)
```

### Deleting and stats

```python
mem.delete(z.id)       # removes zettel and cleans up all links
print(mem.stats)       # {"total_zettels": ..., "total_connections": ..., ...}
```

---

## Embedding Backend

For higher-quality semantic search, swap TF-IDF for real embeddings. Provide any function with the signature `(list[str]) -> np.ndarray`:

### OpenAI

```python
import numpy as np
from openai import OpenAI
from zettelkasten_memory import ZettelMemory, EmbeddingBackend

client = OpenAI()  # uses OPENAI_API_KEY env var

def embed(texts: list[str]) -> np.ndarray:
    resp = client.embeddings.create(input=texts, model="text-embedding-3-small")
    return np.array([e.embedding for e in resp.data])

mem = ZettelMemory(backend=EmbeddingBackend(embed_fn=embed))
mem.add("The project uses FastAPI")
results = mem.search("what web framework?")
```

### Sentence Transformers (local, free)

```python
import numpy as np
from sentence_transformers import SentenceTransformer
from zettelkasten_memory import ZettelMemory, EmbeddingBackend

model = SentenceTransformer("all-MiniLM-L6-v2")

def embed(texts: list[str]) -> np.ndarray:
    return model.encode(texts)

mem = ZettelMemory(backend=EmbeddingBackend(embed_fn=embed))
```

### Cohere, Voyage, or any other provider

```python
# Same pattern — just return an np.ndarray of shape (n_texts, dim)
def embed(texts: list[str]) -> np.ndarray:
    return np.array(your_api_call(texts))

mem = ZettelMemory(backend=EmbeddingBackend(embed_fn=embed))
```

### Persistence with embeddings

```python
# Save works the same way
mem.save("memory.json")

# When loading, pass the same embed_fn (it can't be serialised)
mem = ZettelMemory.load("memory.json", embed_fn=embed)
```

---

## CrewAI Adapter

Drop-in replacement for CrewAI's memory storage:

```python
from crewai import Crew
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

`ZettelStorage` implements `save()`, `search()`, and `reset()` — the interface CrewAI expects. Memories auto-link across tasks, so agents benefit from connections discovered in earlier steps.

---

## LangGraph Adapter

Implements LangGraph's `BaseStore` interface:

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

---

## Claude Code / MCP Server

Run as an MCP server that gives Claude persistent, searchable memory:

```bash
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

---

## How It Works

### Storage

Each zettel stores: content, metadata (arbitrary key-value), auto-extracted tags, importance (0-1), bidirectional connections to other zettels, creation/access timestamps, and an access counter.

### Pluggable Backends

The search engine is abstracted behind a `SearchBackend` protocol. Two backends ship out of the box:

| Backend | How it works | When to use |
|---|---|---|
| `TfidfBackend` (default) | scikit-learn TF-IDF vectors + cosine similarity | Zero config, no API keys, works offline |
| `EmbeddingBackend` | Your embedding function + normalised dot product | Higher quality search, understands synonyms/paraphrases |

Both backends handle: index building, similarity queries, auto-link discovery, and tag extraction.

### Auto-linking

When a new zettel is added, the backend compares it against all existing zettels. Any pair exceeding `connection_threshold` (default 0.25 cosine similarity) gets a bidirectional link.

### Search Ranking

```
score = similarity * importance * (0.7 + 0.3 * recency) * connection_boost
```

- **recency** decays over days since last access
- **connection_boost** rewards well-connected zettels (up to 2x for 10+ links)
- Falls back to keyword overlap when the backend yields no matches

### Eviction

When over capacity, the least-valuable zettels are removed:

```
value = importance * (1 + access_count) * recency
```

### Persistence

JSON file with all zettels, config, and backend type. Load/save are explicit — no background I/O.

---

## Configuration

```python
from zettelkasten_memory import ZettelMemory, TfidfBackend, EmbeddingBackend

# TF-IDF (default)
mem = ZettelMemory(
    max_zettels=5000,              # capacity before eviction
    connection_threshold=0.25,      # min similarity to auto-link
)

# Custom TF-IDF settings
mem = ZettelMemory(
    backend=TfidfBackend(max_features=10000, ngram_range=(1, 3)),
)

# Embeddings
mem = ZettelMemory(
    backend=EmbeddingBackend(embed_fn=my_fn, batch_size=128),
)
```

---

## Development

```bash
pixi install -e dev
pixi run test      # pytest -v tests/
pixi run format    # black src/ tests/
pixi run lint      # py_compile check
```

---

## Roadmap

Here's what's planned for future releases:

- **Hybrid search** — combine TF-IDF keyword matching with embedding similarity for best-of-both-worlds retrieval
- **Async embedding backend** — non-blocking API calls for embedding providers, useful in async agent frameworks
- **Streaming persistence** — incremental writes instead of full JSON dumps, for large memory stores
- **Memory consolidation** — automatically merge near-duplicate zettels and summarise clusters to stay within capacity without losing information
- **Importance decay and reinforcement** — automatically decrease importance of unused memories over time, and boost memories that are frequently retrieved
- **Multi-modal zettels** — support images, code snippets, and structured data as first-class zettel content alongside text
- **Graph visualisation** — export the zettel link graph to formats like DOT/Graphviz or interactive HTML for exploring memory structure
- **Vector database backends** — pluggable storage backends using FAISS, Qdrant, ChromaDB, or Pinecone for scaling beyond in-memory limits
- **Pre-built embedding providers** — out-of-the-box wrappers for OpenAI, Cohere, Voyage, and sentence-transformers so users don't have to write their own `embed_fn`
- **Claude Code memory commands** — higher-level MCP tools like `memory_reflect` (summarise what you know about topic X) and `memory_prune` (clean up stale entries)

---

## License

[MIT](LICENSE) — Kevin Keller ([@KellerKev](https://github.com/KellerKev)) — [kevinkeller.org](https://kevinkeller.org)
