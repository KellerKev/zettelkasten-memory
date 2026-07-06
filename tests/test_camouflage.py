"""Tests for AES-SIV PII camouflage (deterministic tokenization)."""

from __future__ import annotations

import json
import os

import pytest

pytest.importorskip("cryptography")

from zettelkasten_memory import ZettelMemory
from zettelkasten_memory.camouflage import ENV_PII_KEY, CamouflageCodec, CamouflageError

KEY = os.urandom(64)  # AES-256-SIV


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    monkeypatch.delenv(ENV_PII_KEY, raising=False)


def _codec(**kw) -> CamouflageCodec:
    return CamouflageCodec(key=KEY, **kw)


# ------------------------------------------------------------------
# Codec unit tests
# ------------------------------------------------------------------


def test_email_roundtrip():
    codec = _codec()
    text = "reach the admin at admin@example.com for access"
    masked = codec.tokenize(text)
    assert "admin@example.com" not in masked
    assert "[pii-email-" in masked
    assert codec.detokenize(masked) == text


def test_phone_roundtrip():
    codec = _codec()
    text = "call +1 (415) 555-0123 tomorrow"
    masked = codec.tokenize(text)
    assert "555-0123" not in masked
    assert "[pii-phone-" in masked
    assert codec.detokenize(masked) == text


def test_credit_card_roundtrip_luhn_and_last4():
    codec = _codec()
    text = "card on file: 4111 1111 1111 1111 expires soon"
    masked = codec.tokenize(text)
    assert "4111 1111 1111 1111" not in masked
    assert "[pii-credit_card-" in masked
    assert masked.count("-1111]") == 1  # keep_last4 hint
    assert codec.detokenize(masked) == text


def test_luhn_rejects_non_card_numbers():
    codec = _codec()
    text = "order number 1234 5678 9012 3456 shipped"  # fails Luhn
    assert "[pii-credit_card-" not in codec.tokenize(text)


def test_determinism_same_value_same_token():
    codec = _codec()
    a = codec.tokenize("email me: person@corp.io")
    b = codec.tokenize("their address is person@corp.io indeed")
    token_a = [w for w in a.split() if w.startswith("[pii-email-")][0]
    token_b = [w for w in b.split() if w.startswith("[pii-email-")][0]
    assert token_a == token_b


def test_names_wordlist():
    codec = _codec(names={"Ada Lovelace"})
    masked = codec.tokenize("Ada Lovelace wrote the first program")
    assert "Ada Lovelace" not in masked
    assert "[pii-name-" in masked
    assert codec.detokenize(masked) == "Ada Lovelace wrote the first program"


def test_extra_patterns():
    codec = _codec(extra_patterns={"ssn": r"\b\d{3}-\d{2}-\d{4}\b"})
    masked = codec.tokenize("ssn 078-05-1120 on record")
    assert "078-05-1120" not in masked
    assert "[pii-ssn-" in masked
    assert codec.detokenize(masked) == "ssn 078-05-1120 on record"


def test_tokens_not_double_tokenized():
    codec = _codec()
    once = codec.tokenize("mail bob@corp.io now")
    twice = codec.tokenize(once)
    assert once == twice
    assert codec.detokenize(twice) == "mail bob@corp.io now"


def test_reveal_false_passthrough():
    codec = _codec(reveal=False)
    masked = codec.tokenize("mail bob@corp.io")
    assert codec.detokenize(masked) == masked


def test_wrong_key_detokenize_fails():
    masked = _codec().tokenize("mail bob@corp.io")
    other = CamouflageCodec(key=os.urandom(64))
    with pytest.raises(CamouflageError):
        other.detokenize(masked)


def test_missing_key_raises():
    with pytest.raises(CamouflageError, match="ZETTEL_PII_KEY"):
        CamouflageCodec()


def test_env_key(monkeypatch):
    monkeypatch.setenv(ENV_PII_KEY, KEY.hex())
    codec = CamouflageCodec()
    assert codec.detokenize(codec.tokenize("x@y.io")) == "x@y.io"


def test_repr_hides_key():
    assert KEY.hex()[:16] not in repr(_codec())


# ------------------------------------------------------------------
# ZettelMemory integration
# ------------------------------------------------------------------


def test_add_stores_tokenized_search_reveals():
    mem = ZettelMemory(camouflage=_codec())
    z = mem.add("the customer kevin.k@corp.io prefers weekly reports")

    # in-memory store holds no raw PII
    assert "kevin.k@corp.io" not in mem._zettels[z.id].content

    # searching with the RAW email still finds it (query tokenized the same way)
    results = mem.search("kevin.k@corp.io weekly reports")
    assert results
    assert "kevin.k@corp.io" in results[0].zettel.content  # revealed on the way out


def test_shared_entity_auto_links_via_token():
    mem = ZettelMemory(camouflage=_codec(), connection_threshold=0.05)
    a = mem.add("kevin.k@corp.io opened a support ticket about billing")
    b = mem.add("second interaction with kevin.k@corp.io regarding billing refund")
    assert a.id in mem._zettels[b.id].connections or b.id in mem._zettels[a.id].connections


def test_get_and_context_reveal():
    mem = ZettelMemory(camouflage=_codec())
    z = mem.add("contact maria@corp.io about the deployment")
    got = mem.get(z.id)
    assert "maria@corp.io" in got.content
    assert "maria@corp.io" not in mem._zettels[z.id].content  # store stays tokenized
    ctx = mem.get_context("deployment contact")
    assert "maria@corp.io" in ctx


def test_reveal_false_returns_tokens():
    mem = ZettelMemory(camouflage=_codec(reveal=False))
    mem.add("contact maria@corp.io about the deployment")
    results = mem.search("maria@corp.io deployment")
    assert results
    assert "maria@corp.io" not in results[0].zettel.content
    assert "[pii-email-" in results[0].zettel.content


def test_metadata_tokenized():
    mem = ZettelMemory(camouflage=_codec())
    z = mem.add("note", metadata={"owner": "bob@corp.io", "count": 3})
    assert "bob@corp.io" not in json.dumps(mem._zettels[z.id].metadata)
    assert mem.get(z.id).metadata["owner"] == "bob@corp.io"
    assert mem.get(z.id).metadata["count"] == 3


def test_saved_file_contains_no_raw_pii(tmp_path):
    mem = ZettelMemory(camouflage=_codec())
    mem.add("the customer alice@secret.org called about 4111 1111 1111 1111")
    path = tmp_path / "m.json"
    mem.save(path)
    raw = path.read_bytes()
    assert b"alice@secret.org" not in raw
    assert b"4111 1111 1111 1111" not in raw
    assert json.loads(raw)["camouflage"] is True


def test_camouflage_plus_encryption_compose(tmp_path, monkeypatch):
    from zettelkasten_memory.crypto import ENV_KEY, MAGIC

    monkeypatch.setenv(ENV_KEY, os.urandom(32).hex())
    mem = ZettelMemory(camouflage=_codec())
    mem.add("alice@secret.org has account 4111 1111 1111 1111")
    path = tmp_path / "m.bin"
    mem.save(path)
    assert path.read_bytes().startswith(MAGIC)

    loaded = ZettelMemory.load(path, camouflage=_codec())
    results = loaded.search("alice@secret.org account")
    assert results and "alice@secret.org" in results[0].zettel.content


def test_load_without_codec_warns_but_works(tmp_path, caplog):
    import logging

    mem = ZettelMemory(camouflage=_codec())
    mem.add("mail alice@secret.org today")
    path = tmp_path / "m.json"
    mem.save(path)

    with caplog.at_level(logging.WARNING):
        loaded = ZettelMemory.load(path)
    assert any("camouflage" in r.message for r in caplog.records)
    # still searchable via tokenized query from a fresh codec
    query = _codec().tokenize("alice@secret.org")
    assert loaded.search(query)
