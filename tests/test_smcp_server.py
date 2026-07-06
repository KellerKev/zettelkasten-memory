"""Tests for the SMCP v3 server adapter.

Includes a golden wire-compatibility test that re-derives the key schedule and
signature from raw primitives (hashlib/hmac, independent of smcp_protocol) to
guarantee interoperability with existing SMCP v3 clients.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import hmac
import json
import time
import uuid

import pytest

pytest.importorskip("cryptography")
pytest.importorskip("jwt")
websockets = pytest.importorskip("websockets")

from zettelkasten_memory.adapters.smcp_protocol import (
    SMCPCrypto,
    SMCPProtocolError,
    make_message,
    parse_message,
)
from zettelkasten_memory.adapters.smcp_server import SMCPServerConfig, ZettelSMCPServer
from zettelkasten_memory.core import ZettelMemory

SECRET = "test-shared-secret"
API_KEY_A = "api-key-alpha"
API_KEY_B = "api-key-beta"


@pytest.fixture(scope="module")
def crypto() -> SMCPCrypto:
    # PBKDF2 600k is deliberately slow; derive once per module
    return SMCPCrypto(SECRET)


@pytest.fixture()
def server(crypto) -> ZettelSMCPServer:
    config = SMCPServerConfig(
        secret_key=SECRET,
        api_keys={API_KEY_A: "tenant-a", API_KEY_B: "tenant-b"},
        token_ttl=3600,
    )
    srv = ZettelSMCPServer(ZettelMemory(), config)
    srv.crypto = crypto  # reuse module-scoped derivation
    srv._jwt_secret = crypto.derived_jwt_secret
    return srv


def _roundtrip(
    server: ZettelSMCPServer, crypto: SMCPCrypto, msg_type: str, payload: dict, *, encrypt=True
) -> dict:
    """Client-side encode -> server handle -> client-side decode."""
    wire = make_message(crypto, msg_type, payload, encrypt=encrypt)
    response = server._handle_message(parse_message(crypto, json.dumps(wire)))
    return parse_message(crypto, json.dumps(response), max_skew=0)


def _auth(server, crypto, api_key: str) -> str:
    resp = _roundtrip(server, crypto, "auth", {"api_key": api_key})
    assert resp["payload"]["status"] == "success"
    return resp["payload"]["token"]


# ------------------------------------------------------------------
# Golden wire-compat: independent re-derivation from raw primitives
# ------------------------------------------------------------------


def test_wire_compat_key_schedule_and_signature(crypto):
    """Re-derive the v3 key schedule with raw hashlib/HKDF and verify both
    directions of Fernet + HMAC interop with SMCPCrypto."""
    from cryptography.fernet import Fernet
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.kdf.hkdf import HKDF

    master = hashlib.pbkdf2_hmac("sha256", SECRET.encode(), b"malgra-tunnel-v3", 600_000, dklen=32)
    cipher_key = HKDF(
        algorithm=hashes.SHA256(), length=32, salt=None, info=b"malgra-tunnel-v3-cipher"
    ).derive(master)
    mac_key = HKDF(
        algorithm=hashes.SHA256(), length=32, salt=None, info=b"malgra-tunnel-v3-mac"
    ).derive(master)
    reference_fernet = Fernet(base64.urlsafe_b64encode(cipher_key))

    # payload encrypted by the reference schedule decrypts via SMCPCrypto
    token = reference_fernet.encrypt(json.dumps({"hello": "wire"}).encode()).decode()
    assert crypto.decrypt_payload(token) == {"hello": "wire"}
    # and vice versa
    assert json.loads(reference_fernet.decrypt(crypto.encrypt_payload({"x": 1}).encode())) == {
        "x": 1
    }

    # signature: reference computation (id + type + "<n>.0" + canonical payload)
    mid, mtype, secs = "fixed-id", "tool_invoke", 1750000000
    payload = {"b": 2, "a": 1}
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    expected = hmac.new(
        mac_key, f"{mid}{mtype}{secs}.0{canonical}".encode(), hashlib.sha256
    ).hexdigest()
    assert crypto.sign(mid, mtype, float(secs), payload) == expected


def test_reference_client_envelope_accepted(crypto, server):
    """An envelope built exactly like the canonical client is parsed and served."""
    from cryptography.fernet import Fernet
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.kdf.hkdf import HKDF

    master = hashlib.pbkdf2_hmac("sha256", SECRET.encode(), b"malgra-tunnel-v3", 600_000, dklen=32)
    fernet = Fernet(
        base64.urlsafe_b64encode(
            HKDF(
                algorithm=hashes.SHA256(), length=32, salt=None, info=b"malgra-tunnel-v3-cipher"
            ).derive(master)
        )
    )
    mac_key = HKDF(
        algorithm=hashes.SHA256(), length=32, salt=None, info=b"malgra-tunnel-v3-mac"
    ).derive(master)

    nonce = uuid.uuid4().hex
    mid, secs = str(uuid.uuid4()), int(time.time())
    payload = {"client_id": "ref", "protocol_version": "3.0", "nonce": nonce}
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    envelope = {
        "id": mid,
        "type": "handshake",
        "timestamp": float(secs),
        "payload": payload,
        "encrypted": False,
        "signature": hmac.new(
            mac_key, f"{mid}handshake{secs}.0{canonical}".encode(), hashlib.sha256
        ).hexdigest(),
    }

    response = server._handle_message(parse_message(crypto, json.dumps(envelope)))
    decoded = json.loads(fernet.decrypt(response["payload"]["encrypted_data"].encode()))
    assert decoded["client_nonce"] == nonce
    assert decoded["protocol_version"] == "3.0"


# ------------------------------------------------------------------
# Protocol layer
# ------------------------------------------------------------------


def test_tampered_signature_rejected(crypto):
    wire = make_message(crypto, "auth", {"api_key": "x"})
    wire["signature"] = "0" * 64
    with pytest.raises(SMCPProtocolError, match="signature"):
        parse_message(crypto, json.dumps(wire))


def test_tampered_payload_rejected(crypto):
    wire = make_message(crypto, "auth", {"api_key": "x"})
    wire["payload"] = {"encrypted_data": wire["payload"]["encrypted_data"][:-4] + "AAAA"}
    with pytest.raises(SMCPProtocolError):
        parse_message(crypto, json.dumps(wire))


def test_stale_message_rejected(crypto):
    wire = make_message(crypto, "heartbeat", {"x": 1})
    wire_old = dict(wire)
    old_ts = float(int(time.time()) - 4000)
    wire_old["timestamp"] = old_ts
    wire_old["signature"] = crypto.sign(wire["id"], "heartbeat", old_ts, wire["payload"])
    with pytest.raises(SMCPProtocolError, match="stale"):
        parse_message(crypto, json.dumps(wire_old), max_skew=300)
    # skew=0 disables the check
    parse_message(crypto, json.dumps(wire_old), max_skew=0)


def test_wrong_secret_cannot_forge(crypto):
    other = SMCPCrypto("completely-different-secret", kdf_salt="other-salt")
    wire = make_message(other, "auth", {"api_key": "x"})
    with pytest.raises(SMCPProtocolError, match="signature"):
        parse_message(crypto, json.dumps(wire))


# ------------------------------------------------------------------
# Server flow
# ------------------------------------------------------------------


def test_handshake_echoes_nonce(server, crypto):
    nonce = uuid.uuid4().hex
    resp = _roundtrip(
        server,
        crypto,
        "handshake",
        {"client_id": "t", "protocol_version": "3.0", "nonce": nonce},
        encrypt=False,
    )
    assert resp["type"] == "handshake"
    assert resp["payload"]["client_nonce"] == nonce
    assert resp["payload"]["node_id"] == "zettel-memory"


def test_auth_bad_key_rejected(server, crypto):
    resp = _roundtrip(server, crypto, "auth", {"api_key": "wrong"})
    assert resp["type"] == "error"
    assert "Authentication failed" in resp["payload"]["error"]


def test_discovery_requires_token(server, crypto):
    resp = _roundtrip(server, crypto, "capability_discovery", {"token": "garbage"})
    assert resp["type"] == "error"

    token = _auth(server, crypto, API_KEY_A)
    resp = _roundtrip(server, crypto, "capability_discovery", {"token": token})
    caps = resp["payload"]["capabilities"]
    assert set(caps) == {
        "memory_store",
        "memory_search",
        "memory_get",
        "memory_delete",
        "memory_connections",
        "memory_stats",
    }
    assert caps["memory_store"]["auth_required"] is True


def test_tool_invoke_full_cycle(server, crypto):
    token = _auth(server, crypto, API_KEY_A)

    resp = _roundtrip(
        server,
        crypto,
        "tool_invoke",
        {
            "token": token,
            "tool_name": "memory_store",
            "parameters": {"content": "the gateway runs on port 8766", "tags": ["infra"]},
        },
    )
    assert resp["type"] == "tool_response"
    zid = resp["payload"]["result"]["id"]

    resp = _roundtrip(
        server,
        crypto,
        "tool_invoke",
        {
            "token": token,
            "tool_name": "memory_search",
            "parameters": {"query": "gateway port"},
        },
    )
    results = resp["payload"]["result"]
    assert results and results[0]["id"] == zid

    resp = _roundtrip(
        server,
        crypto,
        "tool_invoke",
        {
            "token": token,
            "tool_name": "memory_delete",
            "parameters": {"memory_id": zid},
        },
    )
    assert resp["payload"]["result"]["deleted"] is True


def test_namespace_binding_from_identity(server, crypto):
    token_a = _auth(server, crypto, API_KEY_A)
    token_b = _auth(server, crypto, API_KEY_B)

    _roundtrip(
        server,
        crypto,
        "tool_invoke",
        {
            "token": token_a,
            "tool_name": "memory_store",
            "parameters": {"content": "tenant-a confidential roadmap"},
        },
    )

    resp = _roundtrip(
        server,
        crypto,
        "tool_invoke",
        {
            "token": token_b,
            "tool_name": "memory_search",
            "parameters": {"query": "tenant-a confidential roadmap"},
        },
    )
    assert resp["payload"]["result"] == []

    # client-supplied namespace parameter is ignored
    resp = _roundtrip(
        server,
        crypto,
        "tool_invoke",
        {
            "token": token_b,
            "tool_name": "memory_search",
            "parameters": {"query": "confidential roadmap", "namespace": "tenant-a"},
        },
    )
    assert resp["payload"]["result"] == []


def test_expired_jwt_rejected(crypto):
    config = SMCPServerConfig(secret_key=SECRET, api_keys={API_KEY_A: "tenant-a"}, token_ttl=-10)
    server = ZettelSMCPServer(ZettelMemory(), config)
    server.crypto = crypto
    server._jwt_secret = crypto.derived_jwt_secret

    token = _auth(server, crypto, API_KEY_A)
    resp = _roundtrip(
        server,
        crypto,
        "tool_invoke",
        {
            "token": token,
            "tool_name": "memory_stats",
            "parameters": {},
        },
    )
    assert resp["type"] == "error"
    assert "Unauthorized" in resp["payload"]["error"]


def test_oversized_content_returns_tool_error(server, crypto, monkeypatch):
    server.memory.max_content_bytes = 64
    token = _auth(server, crypto, API_KEY_A)
    resp = _roundtrip(
        server,
        crypto,
        "tool_invoke",
        {
            "token": token,
            "tool_name": "memory_store",
            "parameters": {"content": "x" * 200},
        },
    )
    assert resp["type"] == "error"
    assert "limit" in resp["payload"]["error"]


# ------------------------------------------------------------------
# Config from env
# ------------------------------------------------------------------


def test_config_requires_secret_and_api_key():
    with pytest.raises(SystemExit, match="SECRET_KEY"):
        SMCPServerConfig.from_env({})
    with pytest.raises(SystemExit, match="API_KEY"):
        SMCPServerConfig.from_env({"ZETTEL_SMCP_SECRET_KEY": "s"})


def test_config_multi_tenant_and_prefix_fallback():
    cfg = SMCPServerConfig.from_env(
        {
            "SMCP_SECRET_KEY": "s",  # fallback prefix
            "ZETTEL_SMCP_API_KEYS": "k1=ns1, k2=ns2",
            "ZETTEL_SMCP_PORT": "9999",
        }
    )
    assert cfg.secret_key == "s"
    assert cfg.api_keys == {"k1": "ns1", "k2": "ns2"}
    assert cfg.port == 9999


def test_config_rejects_unsupported_mode():
    with pytest.raises(SystemExit, match="mode"):
        SMCPServerConfig.from_env(
            {
                "ZETTEL_SMCP_SECRET_KEY": "s",
                "ZETTEL_SMCP_API_KEY": "k",
                "SCP_MODE": "enterprise",
            }
        )


# ------------------------------------------------------------------
# End-to-end over a real websocket
# ------------------------------------------------------------------


@pytest.mark.asyncio
async def test_end_to_end_websocket(crypto, tmp_path):
    config = SMCPServerConfig(secret_key=SECRET, api_keys={API_KEY_A: "tenant-a"}, port=0)
    persist = tmp_path / "smcp_memory.json"
    server = ZettelSMCPServer(ZettelMemory(), config, str(persist))
    server.crypto = crypto
    server._jwt_secret = crypto.derived_jwt_secret

    async with websockets.serve(
        server._connection, "127.0.0.1", 0, max_size=config.max_message_bytes
    ) as ws_server:
        port = ws_server.sockets[0].getsockname()[1]

        async with websockets.connect(f"ws://127.0.0.1:{port}") as ws:

            async def rt(msg_type, payload, encrypt=True):
                await ws.send(json.dumps(make_message(crypto, msg_type, payload, encrypt=encrypt)))
                return parse_message(crypto, await ws.recv(), max_skew=0)

            nonce = uuid.uuid4().hex
            hs = await rt("handshake", {"client_id": "e2e", "nonce": nonce}, encrypt=False)
            assert hs["payload"]["client_nonce"] == nonce

            auth = await rt("auth", {"api_key": API_KEY_A})
            token = auth["payload"]["token"]

            caps = await rt("capability_discovery", {"token": token})
            assert "memory_store" in caps["payload"]["capabilities"]

            stored = await rt(
                "tool_invoke",
                {
                    "token": token,
                    "tool_name": "memory_store",
                    "parameters": {"content": "end to end memory over smcp"},
                },
            )
            assert stored["payload"]["status"] == "success"

            found = await rt(
                "tool_invoke",
                {
                    "token": token,
                    "tool_name": "memory_search",
                    "parameters": {"query": "end to end memory"},
                },
            )
            assert found["payload"]["result"]

    assert persist.exists()  # store persisted on write
