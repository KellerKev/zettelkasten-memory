"""A server restarted on a saved store must search with the embedding function it was
configured with, and must not embed what it already has again (adapters/_tools.build_memory)."""

import numpy as np
import pytest

from zettelkasten_memory.adapters import _tools
from zettelkasten_memory.backends import EmbeddingBackend, HybridBackend, TfidfBackend

WORDS = ["invoice", "deploy", "branch", "month", "lunch", "tests"]


class CountingEmbed:
    """A tiny bag-of-words embedding that counts how many texts it was asked to embed."""

    def __init__(self):
        self.texts = 0

    def __call__(self, texts):
        self.texts += len(texts)
        return np.array(
            [[float(t.lower().count(w)) + 0.01 for w in WORDS] for t in texts], dtype=np.float32
        )


@pytest.mark.parametrize("make", [
    lambda fn: HybridBackend(embed_fn=fn),
    lambda fn: EmbeddingBackend(embed_fn=fn),
])
def test_search_works_after_a_restart_without_embedding_everything_again(tmp_path, make):
    path = str(tmp_path / "store.json")
    first = CountingEmbed()
    mem = _tools.build_memory(path, make(first))
    for text in ("the invoice goes out each month", "deploy from the main branch",
                 "lunch is at noon"):
        _tools.store(mem, text, None, 0.5, None, "text", None, namespace="a")
    assert _tools.search(mem, "invoice month", 1, None, namespace="a")
    _tools.persist_memory(mem, path, encrypt=False)

    second = CountingEmbed()
    again = _tools.build_memory(path, make(second))
    hits = _tools.search(again, "which branch do we deploy from", 1, None, namespace="a")
    assert hits and "deploy" in hits[0]["content"]
    assert second.texts == 1                 # the query, and none of the stored memories
    # and it can still take a new memory
    _tools.store(again, "the tests run before a deploy", None, 0.5, None, "text", None, namespace="a")
    assert _tools.search(again, "tests", 1, None, namespace="a")


def test_a_store_saved_with_another_index_is_indexed_again(tmp_path):
    path = str(tmp_path / "store.json")
    mem = _tools.build_memory(path, TfidfBackend())
    _tools.store(mem, "deploy from the main branch", None, 0.5, None, "text", None, namespace="a")
    _tools.persist_memory(mem, path, encrypt=False)
    fn = CountingEmbed()
    again = _tools.build_memory(path, HybridBackend(embed_fn=fn))
    assert isinstance(again._backend, HybridBackend)
    assert _tools.search(again, "deploy branch", 1, None, namespace="a")
    assert fn.texts >= 2                     # the stored memory and the query
