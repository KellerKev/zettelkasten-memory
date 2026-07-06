"""Encryption at rest for persisted memory stores.

Whole-file AES-256-GCM with a small versioned binary envelope.  The envelope is
self-describing and auto-detected on load, so encrypted and legacy plaintext
JSON stores coexist transparently.

Envelope layout::

    offset size  field
    0      4     magic b"ZMEM"
    4      1     format version (0x01)
    5      1     key mode: 0x01 raw 32-byte key, 0x02 scrypt passphrase
    --- mode 0x02 only ---
    6      16    scrypt salt
    22     1     scrypt log2(N)
    23     1     scrypt r
    24     1     scrypt p
    --- always ---
    +0     12    AES-GCM nonce (fresh per write)
    +12    ...   ciphertext || 16-byte GCM tag

The full header (magic through nonce) is authenticated as GCM associated data,
so tampering with any header byte — including downgrading scrypt parameters —
fails decryption.

Key material is resolved from, in order: an explicit argument, the
``ZETTEL_MEMORY_KEY`` env var (hex or base64, 32 bytes), a file named by
``ZETTEL_MEMORY_KEY_FILE``, or the ``ZETTEL_MEMORY_PASSPHRASE`` env var
(scrypt-derived).  Keys are never accepted on the command line and never
included in error messages.

Threat model: this protects store files at rest — on disk, in backups, on
shared or synced volumes.  It does not protect against a compromised process,
memory inspection, or anything that can read the environment; CPython cannot
zeroize key material.

Requires the ``cryptography`` package (extra: ``zettelkasten-memory[crypto]``).
"""

from __future__ import annotations

import base64
import binascii
import os
from pathlib import Path

MAGIC = b"ZMEM"
_FORMAT_VERSION = 1
_MODE_RAW_KEY = 0x01
_MODE_SCRYPT = 0x02
_NONCE_SIZE = 12
_SALT_SIZE = 16
_SCRYPT_LOG2_N = 15
_SCRYPT_R = 8
_SCRYPT_P = 1

ENV_KEY = "ZETTEL_MEMORY_KEY"
ENV_KEY_FILE = "ZETTEL_MEMORY_KEY_FILE"
ENV_PASSPHRASE = "ZETTEL_MEMORY_PASSPHRASE"


class EncryptionError(Exception):
    """Encryption or decryption failed."""


class KeyNotFoundError(EncryptionError):
    """No key material could be resolved."""


def _require_cryptography():
    try:
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM

        return AESGCM
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "the 'cryptography' package is required for encrypted persistence. "
            "Install with: pip install 'zettelkasten-memory[crypto]'"
        ) from exc


def _decode_key_material(text: str) -> bytes:
    """Decode a 32-byte key given as hex or base64 text."""
    text = text.strip()
    try:
        raw = bytes.fromhex(text)
        if len(raw) == 32:
            return raw
    except ValueError:
        pass
    for decoder in (base64.b64decode, base64.urlsafe_b64decode):
        try:
            raw = decoder(text)
            if len(raw) == 32:
                return raw
        except (ValueError, binascii.Error):
            continue
    raise EncryptionError("key material must decode to exactly 32 bytes (hex or base64)")


def resolve_key(key: bytes | str | None = None) -> bytes | None:
    """Resolve a 32-byte AES key from an explicit argument or the environment.

    Returns None when no raw-key material is configured (a passphrase may
    still be available — see ``resolve_passphrase``).
    """
    if key is not None:
        if isinstance(key, bytes):
            if len(key) != 32:
                raise EncryptionError("explicit key must be exactly 32 bytes")
            return key
        return _decode_key_material(key)

    env_key = os.environ.get(ENV_KEY)
    if env_key:
        return _decode_key_material(env_key)

    key_file = os.environ.get(ENV_KEY_FILE)
    if key_file:
        raw = Path(key_file).read_bytes()
        if len(raw) == 32:
            return raw
        return _decode_key_material(raw.decode("utf-8", errors="strict"))

    return None


def resolve_passphrase(passphrase: str | None = None) -> str | None:
    return passphrase if passphrase is not None else os.environ.get(ENV_PASSPHRASE) or None


def encryption_available(key: bytes | str | None = None, passphrase: str | None = None) -> bool:
    """True when any key material (key, key file, or passphrase) is configured."""
    try:
        if resolve_key(key) is not None:
            return True
    except EncryptionError:
        return False
    return resolve_passphrase(passphrase) is not None


def is_encrypted(blob: bytes) -> bool:
    return blob[: len(MAGIC)] == MAGIC


def _derive_scrypt(passphrase: str, salt: bytes, log2_n: int, r: int, p: int) -> bytes:
    import hashlib

    return hashlib.scrypt(
        passphrase.encode("utf-8"),
        salt=salt,
        n=1 << log2_n,
        r=r,
        p=p,
        maxmem=256 * 1024 * 1024,
        dklen=32,
    )


def encrypt_bytes(
    plaintext: bytes,
    *,
    key: bytes | str | None = None,
    passphrase: str | None = None,
) -> bytes:
    """Encrypt *plaintext* into a ZMEM envelope.

    A raw key (explicit or from env) takes precedence over a passphrase.
    """
    AESGCM = _require_cryptography()

    raw_key = resolve_key(key)
    if raw_key is not None:
        header = MAGIC + bytes([_FORMAT_VERSION, _MODE_RAW_KEY])
        aes_key = raw_key
    else:
        phrase = resolve_passphrase(passphrase)
        if phrase is None:
            raise KeyNotFoundError(
                f"no key material: set {ENV_KEY}, {ENV_KEY_FILE}, or {ENV_PASSPHRASE}"
            )
        salt = os.urandom(_SALT_SIZE)
        header = (
            MAGIC
            + bytes([_FORMAT_VERSION, _MODE_SCRYPT])
            + salt
            + bytes([_SCRYPT_LOG2_N, _SCRYPT_R, _SCRYPT_P])
        )
        aes_key = _derive_scrypt(phrase, salt, _SCRYPT_LOG2_N, _SCRYPT_R, _SCRYPT_P)

    nonce = os.urandom(_NONCE_SIZE)
    aad = header + nonce
    ciphertext = AESGCM(aes_key).encrypt(nonce, plaintext, aad)
    return header + nonce + ciphertext


def decrypt_bytes(
    blob: bytes,
    *,
    key: bytes | str | None = None,
    passphrase: str | None = None,
) -> bytes:
    """Decrypt a ZMEM envelope produced by ``encrypt_bytes``."""
    AESGCM = _require_cryptography()
    from cryptography.exceptions import InvalidTag

    if not is_encrypted(blob):
        raise EncryptionError("not an encrypted memory store (missing ZMEM magic)")
    if len(blob) < 6:
        raise EncryptionError("truncated encrypted store")

    version, mode = blob[4], blob[5]
    if version != _FORMAT_VERSION:
        raise EncryptionError(f"unsupported envelope version {version}")

    if mode == _MODE_RAW_KEY:
        header_end = 6
        aes_key = resolve_key(key)
        if aes_key is None:
            raise KeyNotFoundError(
                f"store is encrypted with a raw key: set {ENV_KEY} or {ENV_KEY_FILE}"
            )
    elif mode == _MODE_SCRYPT:
        header_end = 6 + _SALT_SIZE + 3
        if len(blob) < header_end + _NONCE_SIZE:
            raise EncryptionError("truncated encrypted store")
        salt = blob[6 : 6 + _SALT_SIZE]
        log2_n, r, p = blob[6 + _SALT_SIZE : header_end]
        phrase = resolve_passphrase(passphrase)
        if phrase is None:
            raise KeyNotFoundError(f"store is encrypted with a passphrase: set {ENV_PASSPHRASE}")
        if log2_n > 22:
            raise EncryptionError("unreasonable scrypt parameters in envelope")
        aes_key = _derive_scrypt(phrase, salt, log2_n, r, p)
    else:
        raise EncryptionError(f"unknown key mode {mode}")

    header = blob[:header_end]
    nonce = blob[header_end : header_end + _NONCE_SIZE]
    ciphertext = blob[header_end + _NONCE_SIZE :]
    if len(nonce) != _NONCE_SIZE or not ciphertext:
        raise EncryptionError("truncated encrypted store")

    try:
        return AESGCM(aes_key).decrypt(nonce, ciphertext, header + nonce)
    except InvalidTag as exc:
        raise EncryptionError("decryption failed: wrong key or corrupted file") from exc
