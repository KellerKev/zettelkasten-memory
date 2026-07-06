"""Tests for encrypted persistence (AES-256-GCM ZMEM envelope)."""

from __future__ import annotations

import json
import os

import numpy as np
import pytest

pytest.importorskip("cryptography")

from zettelkasten_memory import ZettelMemory
from zettelkasten_memory.backends import EmbeddingBackend
from zettelkasten_memory.crypto import (
    ENV_KEY,
    ENV_PASSPHRASE,
    MAGIC,
    EncryptionError,
    KeyNotFoundError,
    decrypt_bytes,
    encrypt_bytes,
    encryption_available,
    is_encrypted,
    resolve_key,
)

KEY = os.urandom(32)
KEY_HEX = KEY.hex()
OTHER_KEY = os.urandom(32)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for var in (ENV_KEY, "ZETTEL_MEMORY_KEY_FILE", ENV_PASSPHRASE):
        monkeypatch.delenv(var, raising=False)


def _fake_embed(texts):
    out = []
    for t in texts:
        state = np.random.default_rng(abs(hash(t)) % (2**32))
        out.append(state.random(8))
    return np.array(out)


# ------------------------------------------------------------------
# Envelope primitives
# ------------------------------------------------------------------


def test_roundtrip_raw_key():
    blob = encrypt_bytes(b"hello world", key=KEY)
    assert is_encrypted(blob)
    assert decrypt_bytes(blob, key=KEY) == b"hello world"


def test_roundtrip_hex_and_b64_key():
    import base64

    blob = encrypt_bytes(b"data", key=KEY_HEX)
    assert decrypt_bytes(blob, key=base64.b64encode(KEY).decode()) == b"data"


def test_roundtrip_passphrase_scrypt():
    blob = encrypt_bytes(b"secret payload", passphrase="correct horse battery staple")
    assert is_encrypted(blob)
    assert decrypt_bytes(blob, passphrase="correct horse battery staple") == b"secret payload"


def test_wrong_key_fails_cleanly():
    blob = encrypt_bytes(b"data", key=KEY)
    with pytest.raises(EncryptionError) as exc:
        decrypt_bytes(blob, key=OTHER_KEY)
    assert KEY_HEX not in str(exc.value)


def test_tamper_ciphertext_detected():
    blob = bytearray(encrypt_bytes(b"data", key=KEY))
    blob[-1] ^= 0xFF
    with pytest.raises(EncryptionError):
        decrypt_bytes(bytes(blob), key=KEY)


def test_tamper_header_detected():
    blob = bytearray(encrypt_bytes(b"data", passphrase="pw"))
    blob[22] ^= 0x01  # scrypt log2(N) byte — AAD covers the header
    with pytest.raises(EncryptionError):
        decrypt_bytes(bytes(blob), passphrase="pw")


def test_nonce_freshness():
    assert encrypt_bytes(b"same", key=KEY) != encrypt_bytes(b"same", key=KEY)


def test_is_encrypted_vs_plaintext_json():
    assert not is_encrypted(b'{"version": 1}')
    assert is_encrypted(MAGIC + b"anything")


def test_resolve_key_errors_on_bad_length():
    with pytest.raises(EncryptionError):
        resolve_key(b"short")
    with pytest.raises(EncryptionError):
        resolve_key("deadbeef")


def test_resolve_key_from_env(monkeypatch):
    monkeypatch.setenv(ENV_KEY, KEY_HEX)
    assert resolve_key() == KEY
    assert encryption_available()


def test_resolve_key_from_key_file(monkeypatch, tmp_path):
    key_file = tmp_path / "store.key"
    key_file.write_bytes(KEY)
    monkeypatch.setenv("ZETTEL_MEMORY_KEY_FILE", str(key_file))
    assert resolve_key() == KEY


def test_no_key_material_raises():
    with pytest.raises(KeyNotFoundError):
        encrypt_bytes(b"data")


# ------------------------------------------------------------------
# ZettelMemory save/load integration
# ------------------------------------------------------------------


def test_save_auto_plaintext_without_key(tmp_path):
    mem = ZettelMemory()
    mem.add("plain note")
    path = tmp_path / "m.json"
    mem.save(path)
    raw = path.read_bytes()
    assert raw.startswith(b"{")
    assert b"plain note" in raw


def test_save_auto_encrypts_with_env_key(monkeypatch, tmp_path):
    monkeypatch.setenv(ENV_KEY, KEY_HEX)
    mem = ZettelMemory()
    mem.add("secret note about the gateway")
    path = tmp_path / "m.json"
    mem.save(path)
    raw = path.read_bytes()
    assert raw.startswith(MAGIC)
    assert b"secret note" not in raw

    loaded = ZettelMemory.load(path)
    assert any(z.content == "secret note about the gateway" for z in loaded._zettels.values())


def test_save_encrypt_true_without_key_raises(tmp_path):
    mem = ZettelMemory()
    mem.add("note")
    with pytest.raises(KeyNotFoundError):
        mem.save(tmp_path / "m.json", encrypt=True)


def test_load_wrong_key_raises_not_garbage(monkeypatch, tmp_path):
    path = tmp_path / "m.json"
    mem = ZettelMemory()
    mem.add("note")
    mem.save(path, key=KEY)
    with pytest.raises(EncryptionError):
        ZettelMemory.load(path, key=OTHER_KEY)


def test_downgrade_guard(monkeypatch, tmp_path):
    path = tmp_path / "m.json"
    mem = ZettelMemory()
    mem.add("note")
    mem.save(path, key=KEY)

    loaded = ZettelMemory.load(path, key=KEY)
    # no key resolvable now -> refuses silent plaintext downgrade
    with pytest.raises(KeyNotFoundError):
        loaded.save(path)
    # explicit opt-out works (documented decrypt-migration path)
    loaded.save(path, encrypt=False)
    assert path.read_bytes().startswith(b"{")


def test_key_rotation(tmp_path):
    path = tmp_path / "m.json"
    mem = ZettelMemory()
    z = mem.add("rotate me")
    mem.save(path, key=KEY)

    loaded = ZettelMemory.load(path, key=KEY)
    loaded.save(path, key=OTHER_KEY)
    reloaded = ZettelMemory.load(path, key=OTHER_KEY)
    assert z.id in reloaded._zettels
    with pytest.raises(EncryptionError):
        ZettelMemory.load(path, key=KEY)


def test_legacy_plaintext_still_loads(tmp_path):
    path = tmp_path / "legacy.json"
    v1 = {
        "version": 1,
        "zettels": [],
        "config": {"max_zettels": 100, "connection_threshold": 0.25},
        "backend": {"type": "tfidf"},
    }
    path.write_text(json.dumps(v1))
    mem = ZettelMemory.load(path)
    assert mem.max_zettels == 100


def test_encrypted_embedding_vectors_roundtrip(monkeypatch, tmp_path):
    monkeypatch.setenv(ENV_KEY, KEY_HEX)
    backend = EmbeddingBackend(embed_fn=_fake_embed)
    mem = ZettelMemory(backend=backend)
    mem.add("vector note")
    mem.search("vector note")  # builds the index so vectors are persisted
    path = tmp_path / "emb.bin"
    mem.save(path)
    assert path.read_bytes().startswith(MAGIC)
    loaded = ZettelMemory.load(path)  # vectors persisted -> no embed_fn needed to search
    assert loaded.search("vector note", limit=3)


def test_atomic_write_no_tmp_left(tmp_path):
    mem = ZettelMemory()
    mem.add("note")
    path = tmp_path / "m.json"
    mem.save(path)
    assert not list(tmp_path.glob("*.tmp"))
