# zettelkasten-memory

Zettelkasten-inspired semantic memory for AI agents. Works standalone or as a drop-in plugin for **CrewAI**, **LangGraph**, and **Claude Code** (via MCP).

Each memory is a "zettel" (note) that is automatically tagged, scored by importance, and **linked to related memories** by semantic similarity. Search results are ranked by a composite score of text similarity, importance, recency, and graph connectivity — so well-connected, important memories surface first.

---

## Table of Contents

- [Install](#install)
- [Quick Start](#quick-start)
- [Backends](#backends)
- [Embedding Providers](#embedding-providers)
- [Vector Compression](#vector-compression)
- [Claude Code / MCP Server](#claude-code--mcp-server)
- [CrewAI Adapter](#crewai-adapter)
- [LangGraph Adapter](#langgraph-adapter)
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
pixi install -e mcp          # MCP server for Claude Code
pixi install -e embeddings   # OpenAI embeddings support
pixi install -e dev          # Development (pytest, black)

# Provider extras
pip install zettelkasten-memory[providers-cohere]    # Cohere embeddings
pip install zettelkasten-memory[providers-voyage]     # Voyage AI embeddings
pip install zettelkasten-memory[providers-local]      # sentence-transformers (local, free)
pip install zettelkasten-memory[all-providers]        # All providers
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

## Backends

The search engine is abstracted behind a `SearchBackend` protocol. Two backends ship out of the box:

### TF-IDF Backend (default)

Zero-config keyword search using scikit-learn TF-IDF vectors and cosine similarity. No API keys, no network calls, works offline. Good enough for keyword-heavy searches but doesn't understand synonyms or paraphrases.

```python
from zettelkasten_memory import ZettelMemory, TfidfBackend

mem = ZettelMemory()  # TF-IDF is the default
mem = ZettelMemory(backend=TfidfBackend(max_features=10000, ngram_range=(1, 3)))
```

### Embedding Backend

Semantic search powered by real embedding vectors. Understands synonyms, paraphrases, and meaning — "how do we ship the app?" finds "Docker deployment on AWS ECS". Requires an embedding function or a [pre-built provider](#embedding-providers).

```python
from zettelkasten_memory import ZettelMemory, EmbeddingBackend

# Using a pre-built provider (recommended)
backend = EmbeddingBackend.from_provider("ollama")
mem = ZettelMemory(backend=backend)

# Or bring your own function: (list[str]) -> np.ndarray
backend = EmbeddingBackend(embed_fn=my_embed_fn, batch_size=128)
mem = ZettelMemory(backend=backend)
```

**Embedding persistence:** Vectors are saved alongside zettels. Loading a saved memory **does not re-embed** — vectors are restored directly from the file. This means no API calls on startup, and read-only access works without providing `embed_fn`:

```python
mem.save("memory.json")

# Vectors loaded from file — no API calls
mem = ZettelMemory.load("memory.json", embed_fn=embed)  # for adding new memories
mem = ZettelMemory.load("memory.json")                   # read-only, search existing
```

### Backend comparison

| Backend | How it works | Semantic understanding | Config needed |
|---|---|---|---|
| `TfidfBackend` | scikit-learn TF-IDF + cosine similarity | Keyword matching only | None |
| `EmbeddingBackend` | Embedding vectors + normalised dot product | Full semantic search | An embedding provider or function |
| `EmbeddingBackend` + `TurboQuantCompressor` | Same, with compressed storage | Full semantic, <2% recall loss | Provider + `compressor=` flag |

---

## Embedding Providers

Pre-built providers handle API keys, batching, retries, and rate limits out of the box.

```python
from zettelkasten_memory import ZettelMemory, EmbeddingBackend

backend = EmbeddingBackend.from_provider("ollama")     # local, free
backend = EmbeddingBackend.from_provider("openai")     # uses OPENAI_API_KEY
backend = EmbeddingBackend.from_provider("snowflake")  # uses SNOWFLAKE_ACCOUNT + SNOWFLAKE_PAT_TOKEN
backend = EmbeddingBackend.from_provider("cohere")     # uses COHERE_API_KEY
backend = EmbeddingBackend.from_provider("voyage")     # uses VOYAGE_API_KEY
backend = EmbeddingBackend.from_provider("sentence-transformers")  # local, free

mem = ZettelMemory(backend=backend)
```

Or use providers directly as `embed_fn`:

```python
from zettelkasten_memory.providers import OllamaEmbeddings, SnowflakeCortexEmbeddings

embed_fn = OllamaEmbeddings(model="nomic-embed-text")
embed_fn = SnowflakeCortexEmbeddings(account="myorg-myacct", model="snowflake-arctic-embed-l-v2.0")

mem = ZettelMemory(backend=EmbeddingBackend(embed_fn=embed_fn))
```

| Provider | Env Var(s) | Install Extra | Test Status |
|---|---|---|---|
| `OllamaEmbeddings` | *(none — local)* | *(none)* | **Tested** (nomic-embed-text) |
| `OpenAIEmbeddings` | `OPENAI_API_KEY` | `embeddings` | Untested |
| `SnowflakeCortexEmbeddings` | `SNOWFLAKE_ACCOUNT` + `SNOWFLAKE_PAT_TOKEN` | *(none)* | Untested |
| `CohereEmbeddings` | `COHERE_API_KEY` | `providers-cohere` | Untested |
| `VoyageEmbeddings` | `VOYAGE_API_KEY` | `providers-voyage` | Untested |
| `SentenceTransformerEmbeddings` | *(none — local)* | `providers-local` | Untested |

> **Note:** The Ollama provider is the only one with a full end-to-end test suite (real embeddings from `nomic-embed-text`). All other providers implement the same `_BaseProvider` interface and are expected to work but have not been tested against live APIs. The Snowflake Cortex provider calls the `/api/v2/cortex/inference:embed` REST endpoint using a PAT (Programmatic Access Token) — no Snowflake SDK required.

---

## Vector Compression

### Background: TurboQuant

The compression layer is inspired by the **TurboQuant** family of algorithms used in research on KV-cache compression for large language models (papers like [KV-Quant](https://arxiv.org/abs/2401.18079) and [KIVI](https://arxiv.org/abs/2402.02750)). These techniques were originally designed to compress the Key/Value attention tensors inside transformer models to reduce GPU memory during inference.

We apply the **same core algorithms** — PolarQuant and QJL — to a different problem: compressing the **embedding vectors stored in the zettelkasten memory**. The math is identical, but instead of compressing KV tensors on a GPU, we compress embedding vectors in a NumPy array on the CPU.

### How it works

The `TurboQuantCompressor` uses a two-stage approach:

1. **PolarQuant** — A random orthogonal rotation (data-oblivious, no training needed) followed by scalar quantization to 4-bit integers. The rotation spreads information evenly across dimensions so per-dimension quantization loses minimal information.

2. **QJL 1-bit residual** — The quantization error is projected through a random matrix, and only the **sign** of each projection is stored (1 bit each). This captures the direction of the residual error and allows partial correction during search.

Queries stay at **full precision**. The search uses asymmetric dot product — the full-precision query is compared against the compressed database vectors. This preserves ranking quality with measured **<2% recall loss** in our test suite.

### What this means in practice

- **Storage:** Embedding vectors go from `n × dim × 4 bytes` (float32) down to roughly `n × dim × 1 byte` (4-bit codes) plus small overhead for residuals and scales. For 768-dim vectors (nomic-embed-text), that's roughly 3-4x smaller files.
- **Startup:** Compressed vectors load faster from disk — less data to read.
- **Search quality:** Nearly identical ranking to full-precision search.
- **No training:** Works on any vectors from any model. No codebook fitting, no calibration data.

### This is NOT an LLM KV cache

To be clear: this does not compress the KV cache inside an LLM's transformer layers. That happens inside the inference engine (vLLM, llama.cpp, etc.) and requires GPU-level integration. What we do is apply the same mathematical techniques to compress the zettelkasten's stored embedding vectors — a much simpler problem that runs in pure NumPy.

### Usage

```python
from zettelkasten_memory import ZettelMemory, EmbeddingBackend, TurboQuantCompressor

compressor = TurboQuantCompressor()           # defaults: 4-bit, 64 projection dims
compressor = TurboQuantCompressor(n_bits=4, proj_dim=128, seed=42)  # custom

# Attach to any embedding backend
backend = EmbeddingBackend.from_provider("ollama", compressor=compressor)
# or
backend = EmbeddingBackend(embed_fn=my_embed_fn, compressor=compressor)

mem = ZettelMemory(backend=backend)
mem.add("The project uses FastAPI")
mem.save("memory.json")  # compressed vectors persisted — much smaller file
```

Via the MCP server:

```bash
python -m zettelkasten_memory.adapters.mcp_server \
    --persist memory.json --provider ollama --compress
```

---

## Claude Code / MCP Server

Run as an MCP server that gives Claude persistent, searchable memory:

```bash
# TF-IDF (default — zero config)
pixi run -e mcp python -m zettelkasten_memory.adapters.mcp_server --persist memory.json

# With Ollama embeddings (local, free)
pixi run -e mcp python -m zettelkasten_memory.adapters.mcp_server \
    --persist memory.json --provider ollama

# With Ollama + compression
pixi run -e mcp python -m zettelkasten_memory.adapters.mcp_server \
    --persist memory.json --provider ollama --compress

# With OpenAI
pixi run -e mcp python -m zettelkasten_memory.adapters.mcp_server \
    --persist memory.json --provider openai

# With Snowflake Cortex
pixi run -e mcp python -m zettelkasten_memory.adapters.mcp_server \
    --persist memory.json --provider snowflake --account myorg-myaccount
```

Configure in `~/.claude/settings.json`:

```json
{
  "mcpServers": {
    "zettel-memory": {
      "command": "pixi",
      "args": ["run", "-e", "mcp", "python", "-m",
               "zettelkasten_memory.adapters.mcp_server",
               "--persist", "/path/to/memory.json",
               "--provider", "ollama"],
      "cwd": "/path/to/zettelkasten-memory"
    }
  }
}
```

Server flags:

| Flag | Description |
|---|---|
| `--persist PATH` | JSON file for persistence |
| `--provider NAME` | `tfidf`, `ollama`, `openai`, `cohere`, `voyage`, `snowflake` |
| `--model NAME` | Model name for the chosen provider |
| `--token TOKEN` | API key / PAT token (overrides env var) |
| `--account ACCT` | Snowflake account identifier (`orgname-accountname`) |
| `--base-url URL` | Ollama base URL (default: `http://localhost:11434`) |
| `--compress` | Enable TurboQuant vector compression |

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

## How It Works

### Storage

Each zettel stores: content, metadata (arbitrary key-value), auto-extracted tags, importance (0-1), bidirectional connections to other zettels, creation/access timestamps, and an access counter.

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

JSON file with all zettels, config, backend type, and embedding vectors. When using `EmbeddingBackend`, vectors are persisted (as float16 or compressed via TurboQuant) so loading doesn't require re-embedding. Load/save are explicit — no background I/O.

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

# Embeddings with provider
mem = ZettelMemory(
    backend=EmbeddingBackend.from_provider("ollama"),
)

# Embeddings with compression
from zettelkasten_memory import TurboQuantCompressor
mem = ZettelMemory(
    backend=EmbeddingBackend.from_provider("ollama", compressor=TurboQuantCompressor()),
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

Tests require a running Ollama instance with `nomic-embed-text`:

```bash
ollama pull nomic-embed-text
pixi run test
```

---

## Roadmap

Here's what's planned for future releases:

- ~~**Pre-built embedding providers**~~ ✅ — OpenAI, Cohere, Voyage, sentence-transformers, Ollama, Snowflake Cortex
- ~~**Vector compression**~~ ✅ — TurboQuant (PolarQuant + QJL) for 3-8x smaller stored vectors
- ~~**Embedding persistence**~~ ✅ — vectors saved/loaded without re-embedding
- ~~**MCP server provider support**~~ ✅ — `--provider`, `--token`, `--compress` flags
- **Hybrid search** — combine TF-IDF keyword matching with embedding similarity for best-of-both-worlds retrieval
- **Async embedding backend** — non-blocking API calls for embedding providers, useful in async agent frameworks
- **Streaming persistence** — incremental writes instead of full JSON dumps, for large memory stores
- **Memory consolidation** — automatically merge near-duplicate zettels and summarise clusters to stay within capacity without losing information
- **Importance decay and reinforcement** — automatically decrease importance of unused memories over time, and boost memories that are frequently retrieved
- **Multi-modal zettels** — support images, code snippets, and structured data as first-class zettel content alongside text
- **Graph visualisation** — export the zettel link graph to formats like DOT/Graphviz or interactive HTML for exploring memory structure
- **Vector database backends** — pluggable storage backends using FAISS, Qdrant, ChromaDB, or Pinecone for scaling beyond in-memory limits
- **Claude Code memory commands** — higher-level MCP tools like `memory_reflect` (summarise what you know about topic X) and `memory_prune` (clean up stale entries)

---

## License

[MIT](LICENSE) — Kevin Keller ([@KellerKev](https://github.com/KellerKev)) — [kevinkeller.org](https://kevinkeller.org)
