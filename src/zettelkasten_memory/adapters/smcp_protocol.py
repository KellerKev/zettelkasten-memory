"""SMCP v3 wire protocol — byte-compatible with the canonical implementation.

SMCP (Secure Model Context Protocol) is a security-hardened MCP variant used
across the surrounding agent ecosystem (terminal agents, LLM gateways, remote
executors).  This module implements the v3 envelope exactly as speced by the
reference implementation, so any existing SMCP client can talk to the zettel
memory server without modification:

- **Key derivation**: ``master = PBKDF2-HMAC-SHA256(secret, salt, 600k)``;
  HKDF splits independent Fernet-cipher and HMAC keys (the raw secret is never
  used as a key).  Default salt ``malgra-tunnel-v3``.
- **Envelope**: ``{id, type, timestamp, payload, encrypted, signature}``;
  encrypted payloads ride as ``{"encrypted_data": <fernet token>}``.
- **Signature**: HMAC-SHA256 over ``id + type + ts + canonical(payload)``
  where ``canonical`` is sorted-keys compact JSON and integer-second
  timestamps render as ``"<n>.0"`` (Rust f64 compat).  Signing happens AFTER
  encryption, so verification runs before any decryption.

Message flow: ``handshake`` (plaintext, nonce echoed for mutual auth) →
``auth`` (api_key → JWT) → ``capability_discovery`` → ``tool_invoke``.

Requires ``cryptography`` (extra: ``zettelkasten-memory[smcp]``).
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import time
import uuid
from typing import Any

PROTOCOL_VERSION = "3.0"
PBKDF2_ITERS = 600_000
DEFAULT_KDF_SALT = b"malgra-tunnel-v3"
HKDF_INFO_CIPHER = b"malgra-tunnel-v3-cipher"
HKDF_INFO_MAC = b"malgra-tunnel-v3-mac"
HKDF_INFO_JWT = b"zettel-smcp-v3-jwt"

MESSAGE_TYPES = frozenset(
    {
        "handshake",
        "auth",
        "capability_discovery",
        "tool_invoke",
        "tool_response",
        "error",
        "heartbeat",
    }
)


class SMCPProtocolError(Exception):
    """Malformed, unauthenticated, or stale message."""


def _hkdf(master: bytes, info: bytes) -> bytes:
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.kdf.hkdf import HKDF

    return HKDF(algorithm=hashes.SHA256(), length=32, salt=None, info=info).derive(master)


class SMCPCrypto:
    """Per-connection crypto context (derive once — PBKDF2 is deliberately slow)."""

    def __init__(self, secret_key: str, kdf_salt: str = "") -> None:
        from cryptography.fernet import Fernet

        salt = kdf_salt.encode() if kdf_salt else DEFAULT_KDF_SALT
        master = hashlib.pbkdf2_hmac("sha256", secret_key.encode(), salt, PBKDF2_ITERS, dklen=32)
        self._fernet = Fernet(base64.urlsafe_b64encode(_hkdf(master, HKDF_INFO_CIPHER)))
        self._mac_key = _hkdf(master, HKDF_INFO_MAC)
        self._jwt_secret = base64.urlsafe_b64encode(_hkdf(master, HKDF_INFO_JWT)).decode()

    @property
    def derived_jwt_secret(self) -> str:
        """Deterministic value derived from the shared channel secret.

        SECURITY: do NOT use this to sign auth JWTs. Every client holds the
        channel secret, so any client can recompute this value and forge a
        token for any namespace. The server signs JWTs with an independent
        secret (``ZETTEL_SMCP_JWT_SECRET`` or a random per-process key). This
        property is retained only for wire-compat tooling and tests that assert
        such a forged token is rejected.
        """
        return self._jwt_secret

    # -- payload crypto -------------------------------------------------

    def encrypt_payload(self, payload: dict[str, Any]) -> str:
        return self._fernet.encrypt(json.dumps(payload).encode()).decode()

    def decrypt_payload(self, encrypted: str) -> dict[str, Any]:
        return json.loads(self._fernet.decrypt(encrypted.encode()))

    # -- envelope signing ------------------------------------------------

    @staticmethod
    def canonical(payload: Any) -> str:
        return json.dumps(payload, sort_keys=True, separators=(",", ":"))

    @staticmethod
    def ts_str(ts: float) -> str:
        f = float(ts)
        return f"{int(f)}.0" if f.is_integer() else str(f)

    def sign(self, msg_id: str, msg_type: str, timestamp: float, wire_payload: Any) -> str:
        data = f"{msg_id}{msg_type}{self.ts_str(timestamp)}{self.canonical(wire_payload)}"
        return hmac.new(self._mac_key, data.encode(), hashlib.sha256).hexdigest()

    def verify(self, message: dict[str, Any]) -> bool:
        signature = message.get("signature")
        if not signature:
            return False
        try:
            timestamp = float(message.get("timestamp", 0))
        except (TypeError, ValueError):
            return False
        expected = self.sign(
            str(message.get("id", "")),
            str(message.get("type", "")),
            timestamp,
            message.get("payload"),
        )
        return hmac.compare_digest(expected, str(signature))


class ReplayGuard:
    """Rejects a signed message whose id was already seen inside the window.

    The signature/freshness checks alone let a captured, still-fresh envelope
    replay verbatim (e.g. a ``tool_invoke:memory_delete``).  One guard per
    connection remembers recently-seen message ids until they age out of the
    freshness window, so a duplicate id is rejected.  Memory is bounded: ids
    are pruned once older than ``ttl``.
    """

    def __init__(self, ttl: float = 300.0) -> None:
        self._ttl = ttl
        self._seen: dict[str, float] = {}

    def check(self, msg_id: str, now: float) -> bool:
        """Return True if *msg_id* is fresh; record it. False if a replay."""
        if self._seen:
            for k in [k for k, exp in self._seen.items() if exp <= now]:
                del self._seen[k]
        if not msg_id:
            return False  # an unidentified message cannot be de-duplicated
        if msg_id in self._seen:
            return False
        self._seen[msg_id] = now + self._ttl
        return True


def make_message(
    crypto: SMCPCrypto,
    msg_type: str,
    payload: dict[str, Any],
    *,
    encrypt: bool = True,
) -> dict[str, Any]:
    """Build a signed (and optionally encrypted) v3 envelope dict."""
    msg_id = str(uuid.uuid4())
    timestamp = float(int(time.time()))  # integer seconds -> signed ts renders "<n>.0"
    wire_payload: Any = payload
    if encrypt and payload:
        wire_payload = {"encrypted_data": crypto.encrypt_payload(payload)}
    return {
        "id": msg_id,
        "type": msg_type,
        "timestamp": timestamp,
        "payload": wire_payload,
        "encrypted": bool(encrypt and payload),
        "signature": crypto.sign(msg_id, msg_type, timestamp, wire_payload),
    }


def parse_message(
    crypto: SMCPCrypto,
    raw: str | bytes,
    *,
    max_skew: float = 300.0,
    replay_guard: "ReplayGuard | None" = None,
) -> dict[str, Any]:
    """Verify, decrypt, and freshness-check an incoming envelope.

    Returns the envelope dict with ``payload`` replaced by the decrypted
    payload.  Verification order matters: signature first (over the wire
    payload), then decryption, then timestamp freshness, then replay.
    ``max_skew=0`` disables the freshness check (the reference implementation
    does not enforce one; this is an accept-side hardening knob).  Pass a
    per-connection *replay_guard* to reject a re-submitted (still-fresh)
    envelope by message id.
    """
    try:
        message = json.loads(raw)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise SMCPProtocolError("invalid json") from exc
    if not isinstance(message, dict):
        raise SMCPProtocolError("invalid message")
    if message.get("type") not in MESSAGE_TYPES:
        raise SMCPProtocolError("unknown message type")

    if not crypto.verify(message):
        raise SMCPProtocolError("invalid signature")

    if message.get("encrypted") and isinstance(message.get("payload"), dict):
        encrypted = message["payload"].get("encrypted_data")
        if not isinstance(encrypted, str):
            raise SMCPProtocolError("malformed encrypted payload")
        try:
            message["payload"] = crypto.decrypt_payload(encrypted)
        except Exception as exc:
            raise SMCPProtocolError("decryption failed") from exc

    if max_skew or replay_guard is not None:
        try:
            ts = float(message.get("timestamp", 0))
        except (TypeError, ValueError) as exc:
            raise SMCPProtocolError("invalid timestamp") from exc
        now = time.time()
        if max_skew and abs(now - ts) > max_skew:
            raise SMCPProtocolError("stale message (timestamp outside accepted window)")
        if replay_guard is not None and not replay_guard.check(str(message.get("id", "")), now):
            raise SMCPProtocolError("replayed or unidentified message")

    return message
