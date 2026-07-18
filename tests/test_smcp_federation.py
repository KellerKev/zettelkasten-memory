"""SMCP A2A federation port (receiver side) — conformance + server dispatch.

``federation_conformance_vectors.json`` is the same cross-language contract the
Python reference (smcp) and Rust (malgra) verify against. This proves zettel's
self-contained port reproduces proof canonicalization/signing, ECDH session
derivation, AES-GCM (session-id AAD) framing, PS256 proof verification, and RS256
client-token verification byte-for-byte, and that the ZettelSMCPServer answers the
two federation verbs over the real signed/encrypted tool_invoke channel.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("cryptography")
pytest.importorskip("jwt")

from zettelkasten_memory.adapters import smcp_federation as fed
from zettelkasten_memory.adapters.smcp_protocol import (
    SMCPCrypto,
    make_message,
    parse_message,
)
from zettelkasten_memory.adapters.smcp_server import SMCPServerConfig, ZettelSMCPServer
from zettelkasten_memory.core import ZettelMemory

VECTORS = json.loads((Path(__file__).parent / "federation_conformance_vectors.json").read_text())

SECRET = "zettel-federation-secret-32-bytes-x"
API_KEY = "fed-api-key"


def test_proof_canonical_and_hmac_vector():
    proof = {
        "client_jwt": "CJWT", "forwarded_by": "nodeA", "forwarded_at": 1700000000.0,
        "task_hash": "abc", "forwarded_to": "nodeB", "nonce": "nonce-1", "expires_at": 1700000300.0,
    }
    canonical = fed.canonical_proof(proof)
    assert canonical == VECTORS["proof_canonical_message"]
    secret = VECTORS["proof_hmac"]["secret"]
    assert fed.hmac_sign_proof(secret, canonical) == VECTORS["proof_hmac"]["signature"]


def test_ecdh_session_vector():
    from cryptography.hazmat.primitives.asymmetric import ec
    e = VECTORS["ecdh"]
    a = ec.derive_private_key(int(e["priv_a_scalar_hex"], 16), ec.SECP256R1())
    key = fed.derive_ecdh_session(a, bytes.fromhex(e["pub_b_x962_hex"]), e["node_a"], e["node_b"])
    assert key.hex() == e["session_key_hex"]


def test_gcm_aad_vector():
    g = VECTORS["gcm_aad"]
    ct_hex, tag_hex = fed.encrypt_session(
        bytes.fromhex(g["key_hex"]), bytes.fromhex(g["nonce_hex"]),
        g["plaintext_utf8"].encode(), g["session_id"])
    assert ct_hex == g["ciphertext_hex"]
    assert tag_hex == g["tag_hex"]


def test_ps256_and_rs256_vectors():
    p = VECTORS["ps256"]
    assert fed.verify_ps256_proof(p["public_key_pem"], p["canonical"], p["signature_hex"])
    r = VECTORS["rs256_token"]
    claims = fed.verify_rs256_token(r["public_key_pem"], r["token"], r["issuer"], r["audience"])
    assert claims["user"] == r["user"]


def test_validator_binding_and_replay():
    v = fed.ProofValidator("nodeB", "sekret")

    def signed(target, signer, nonce):
        proof = {
            "client_jwt": "CJWT", "forwarded_by": signer, "forwarded_at": 1700000000.0,
            "task_hash": "abc", "forwarded_to": target, "nonce": nonce, "expires_at": 4102444800.0,
        }
        return {"proof": proof, "signature": fed.hmac_sign_proof("sekret", fed.canonical_proof(proof)),
                "sig_alg": "HS256"}

    assert v.verify(signed("nodeB", "nodeA", "n-1"), "nodeA")
    with pytest.raises(ValueError):
        v.verify(signed("nodeC", "nodeA", "n-2"), "nodeA")     # wrong target
    with pytest.raises(ValueError):
        v.verify(signed("nodeB", "nodeA", "n-3"), "attacker")  # from_node mismatch
    s = signed("nodeB", "nodeA", "n-4")
    assert v.verify(s, "nodeA")
    with pytest.raises(ValueError):
        v.verify(s, "nodeA")                                   # replay


def _fed_server(crypto):
    config = SMCPServerConfig(
        secret_key=SECRET, api_keys={API_KEY: "tenant"}, node_id="nodeB",
        federation_enabled=True, federation_hmac_secret="fed-shared-secret")
    srv = ZettelSMCPServer(ZettelMemory(), config)
    srv.crypto = crypto
    srv._jwt_secret = crypto.derived_jwt_secret
    return srv


def test_server_dispatch_federation_end_to_end():
    """A federation-enabled zettel node answers federated_key_exchange +
    federated_forward over the real signed/encrypted tool_invoke channel."""
    crypto = SMCPCrypto(SECRET)
    server = _fed_server(crypto)
    conn_fed = server._new_fed_receiver()

    # Authenticate to obtain a session token.
    auth_resp = parse_message(crypto, json.dumps(
        server._handle_message(parse_message(crypto, json.dumps(
            make_message(crypto, "auth", {"api_key": API_KEY}))))))
    token = auth_resp["payload"]["token"]

    def invoke_sync(tool_name, **params):
        wire = make_message(crypto, "tool_invoke",
                            {"token": token, "tool_name": tool_name, "parameters": params})
        resp = server._handle_message(parse_message(crypto, json.dumps(wire)), fed=conn_fed)
        out = parse_message(crypto, json.dumps(resp))
        if out["type"] == "error":
            raise RuntimeError(out["payload"].get("error"))
        return out["payload"]["result"]

    # Sender side (in-process): ECDH then signed+encrypted federated_forward.
    ephemeral, my_pub_hex = fed.initiator_ecdh_start()
    ke = invoke_sync("federated_key_exchange", peer_node="nodeA", peer_pub_hex=my_pub_hex)
    key = fed.initiator_ecdh_finish(ephemeral, ke["peer_pub_hex"], "nodeA", "nodeB")
    task = {"type": "storage", "task_id": "t1"}
    signed = fed.build_signed_proof("unused", task, "nodeB", "nodeA", hmac_secret="fed-shared-secret")
    payload = {"task": task, "auth_proof": signed,
               "forwarding_metadata": {"forwarding_path": ["nodeA"], "task_id": "t1"}}
    enc = fed.encrypt_request(key, fed.session_id_for("nodeA", "nodeB"), payload)
    result = invoke_sync("federated_forward", from_node="nodeA", encrypted_request=enc)

    assert result["status"] == "success"
    assert result["processed_by"] == "nodeB"
    assert result["task_type"] == "storage"


def test_federation_disabled_rejects():
    crypto = SMCPCrypto(SECRET)
    config = SMCPServerConfig(secret_key=SECRET, api_keys={API_KEY: "tenant"}, node_id="nodeB")
    server = ZettelSMCPServer(ZettelMemory(), config)
    server.crypto = crypto
    server._jwt_secret = crypto.derived_jwt_secret
    auth_resp = parse_message(crypto, json.dumps(
        server._handle_message(parse_message(crypto, json.dumps(
            make_message(crypto, "auth", {"api_key": API_KEY}))))))
    token = auth_resp["payload"]["token"]
    wire = make_message(crypto, "tool_invoke",
                        {"token": token, "tool_name": "federated_key_exchange",
                         "parameters": {"peer_node": "nodeA", "peer_pub_hex": "00"}})
    resp = server._handle_message(parse_message(crypto, json.dumps(wire)), fed=None)
    out = parse_message(crypto, json.dumps(resp))
    assert out["type"] == "error"
    assert "federation not enabled" in out["payload"]["error"]
