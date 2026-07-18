"""Self-contained SMCP A2A federation port for zettel (receiver side).

This is the Python counterpart to malgra's ``crates/malgra-tunnel/src/federation.rs``
and SMCP's ``smcp_federated_auth.py`` — the interoperable primitives that let a
zettel memory node participate in an SMCP federation as a *receiver* peer (verify
forwarded, token-authorized requests). The module also carries the sender helpers
(``forward_request``/``build_signed_proof``) so the port stays byte-identical to
rixi's, but a memory server only needs the receiver path:

  * forwarding-proof canonicalization + signing/verification (HMAC "HS256" and
    RSA-PSS "PS256", salt=32 per RFC 7518, with algorithm pinning),
  * forward-secret ECDH (P-256) session-key derivation bound to the exchange
    transcript,
  * AES-256-GCM session encryption with the ``session_id`` bound as AAD,
  * RS256 federation client-token verification (issuer/audience pinned).

The cross-language contract is ``smcp/tests/federation_conformance_vectors.json``;
``agent/tests/test_smcp_federation.py`` asserts byte-for-byte agreement with it.

Federation verbs travel over the normal SMCP ``tool_invoke`` channel as the tool
names ``federated_key_exchange`` (ECDH setup) and ``federated_forward`` (a
token-authorized cross-node request), exactly as malgra's server dispatches them.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import secrets
import time
import uuid
from typing import Any, Awaitable, Callable, Optional

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec, padding
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives.kdf.hkdf import HKDF

# Federation-wide issuer/audience binding. A client token minted by the same
# signer for a *different* service cannot be replayed into the federation.
FEDERATION_ISSUER = "smcp-federation"
FEDERATION_AUDIENCE = "smcp-federation"

_ENC = serialization.Encoding.X962
_FMT = serialization.PublicFormat.UncompressedPoint


# ───────────────────────── proof canonicalization + signing ────────────────
def canonical_proof(proof: dict) -> str:
    """Canonical bytes of a forwarding-proof object: sorted keys, compact
    separators — identical to serde_json's compact output over a sorted map and
    to ``json.dumps(proof, sort_keys=True, separators=(",", ":"))``."""
    return json.dumps(proof, sort_keys=True, separators=(",", ":"))


def hmac_sign_proof(secret: str, canonical: str) -> str:
    return hmac.new(secret.encode(), canonical.encode(), hashlib.sha256).hexdigest()


def hmac_verify_proof(secret: str, canonical: str, signature_hex: str) -> bool:
    try:
        return hmac.compare_digest(hmac_sign_proof(secret, canonical), signature_hex)
    except Exception:
        return False


def _load_private(pem) -> Any:
    return serialization.load_pem_private_key(pem.encode() if isinstance(pem, str) else pem, password=None)


def _load_public(pem) -> Any:
    return serialization.load_pem_public_key(pem.encode() if isinstance(pem, str) else pem)


def sign_ps256_proof(private_key_pem, canonical: str) -> str:
    """Sign a canonical proof with RSA-PSS (PS256, salt length = 32 per RFC 7518),
    hex-encoded. Peers verify with :func:`verify_ps256_proof`."""
    key = _load_private(private_key_pem)
    sig = key.sign(
        canonical.encode(),
        padding.PSS(mgf=padding.MGF1(hashes.SHA256()), salt_length=hashes.SHA256.digest_size),
        hashes.SHA256(),
    )
    return sig.hex()


def verify_ps256_proof(public_key_pem, canonical: str, signature_hex: str) -> bool:
    try:
        pub = _load_public(public_key_pem)
        pub.verify(
            bytes.fromhex(signature_hex),
            canonical.encode(),
            padding.PSS(mgf=padding.MGF1(hashes.SHA256()), salt_length=padding.PSS.MAX_LENGTH),
            hashes.SHA256(),
        )
        return True
    except (InvalidSignature, ValueError):
        return False


# ───────────────────────── forward-secret ECDH (P-256) ─────────────────────
def _pub_bytes(public_key) -> bytes:
    return public_key.public_bytes(encoding=_ENC, format=_FMT)


def derive_ecdh_session(my_private, peer_pub_bytes: bytes, node_id: str, peer_node_id: str) -> bytes:
    """Derive a 32-byte forward-secret session key from a P-256 ECDH exchange:
    HKDF-SHA256 over the shared secret, salt bound to the transcript (SHA-256 of
    the two ephemeral public keys, sorted) and ``info`` = the sorted node pair."""
    peer_pub = ec.EllipticCurvePublicKey.from_encoded_point(ec.SECP256R1(), peer_pub_bytes)
    shared = my_private.exchange(ec.ECDH(), peer_pub)
    my_pub_bytes = _pub_bytes(my_private.public_key())
    transcript = b"".join(sorted([my_pub_bytes, peer_pub_bytes]))
    salt = hashlib.sha256(transcript).digest()
    node_a, node_b = sorted([node_id, peer_node_id])
    info = f"smcp-ecdh-session:{node_a}:{node_b}".encode()
    return HKDF(algorithm=hashes.SHA256(), length=32, salt=salt, info=info).derive(shared)


def perform_ecdh_exchange(peer_pub_hex: str, node_id: str, peer_node_id: str) -> tuple[str, bytes]:
    """Receiver side: given the peer's ephemeral public key (hex, SEC1/X9.62),
    generate our own ephemeral keypair, derive + return ``(our_pub_hex, key)``.
    The ephemeral private key is dropped here (forward secrecy)."""
    ephemeral = ec.generate_private_key(ec.SECP256R1())
    key = derive_ecdh_session(ephemeral, bytes.fromhex(peer_pub_hex), node_id, peer_node_id)
    return _pub_bytes(ephemeral.public_key()).hex(), key


def initiator_ecdh_start() -> tuple[Any, str]:
    """Initiator side, start step: generate an ephemeral keypair, returning
    ``(ephemeral_private, my_pub_hex)``. Send ``my_pub_hex`` to the peer's
    ``federated_key_exchange``; keep the private object for the finish step."""
    ephemeral = ec.generate_private_key(ec.SECP256R1())
    return ephemeral, _pub_bytes(ephemeral.public_key()).hex()


def initiator_ecdh_finish(ephemeral_private, peer_pub_hex: str, node_id: str, peer_node_id: str) -> bytes:
    return derive_ecdh_session(ephemeral_private, bytes.fromhex(peer_pub_hex), node_id, peer_node_id)


def session_id_for(node_id: str, peer_node_id: str) -> str:
    a, b = sorted([node_id, peer_node_id])
    return f"{a}:{b}"


# ───────────────────────── AES-256-GCM session encryption ──────────────────
def encrypt_session(key: bytes, nonce: bytes, plaintext: bytes, session_id: str) -> tuple[str, str]:
    """AES-256-GCM encrypt, binding ``session_id`` as AAD. Returns (ct_hex, tag_hex)."""
    cipher = Cipher(algorithms.AES(key), modes.GCM(nonce))
    enc = cipher.encryptor()
    enc.authenticate_additional_data(session_id.encode())
    ct = enc.update(plaintext) + enc.finalize()
    return ct.hex(), enc.tag.hex()


def decrypt_session(key: bytes, nonce: bytes, ciphertext: bytes, tag: bytes, session_id: str) -> bytes:
    cipher = Cipher(algorithms.AES(key), modes.GCM(nonce, tag))
    dec = cipher.decryptor()
    dec.authenticate_additional_data(session_id.encode())
    return dec.update(ciphertext) + dec.finalize()


def encrypt_request(key: bytes, session_id: str, payload: dict, now: Optional[float] = None) -> dict:
    """Encrypt a forwarded-request payload into the wire ``encrypted_request``
    shape ``{encrypted_data, nonce, tag, session_id, encrypted_at}`` (all hex)."""
    nonce = secrets.token_bytes(12)
    ct_hex, tag_hex = encrypt_session(key, nonce, json.dumps(payload).encode(), session_id)
    return {
        "encrypted_data": ct_hex,
        "nonce": nonce.hex(),
        "tag": tag_hex,
        "session_id": session_id,
        "encrypted_at": time.time() if now is None else now,
    }


# ───────────────────────── RS256 client-token verification ──────────────────
def verify_rs256_token(public_key_pem, token: str, issuer: str = FEDERATION_ISSUER,
                       audience: str = FEDERATION_AUDIENCE) -> dict:
    """Verify an RS256 federation client token against the issuer's PEM public
    key, enforcing issuer/audience + exp/iat. Algorithm-pinned (no HS256
    downgrade). Returns the decoded claims; raises on failure."""
    import jwt

    return jwt.decode(
        token,
        public_key_pem.decode() if isinstance(public_key_pem, bytes) else public_key_pem,
        algorithms=["RS256"],
        audience=audience,
        issuer=issuer,
        options={"require": ["exp", "iat", "aud", "iss"]},
    )


# ───────────────────────── stateful proof validator ────────────────────────
class ProofValidator:
    """Stateful forwarding-proof verifier — the counterpart to SMCP's
    ``verify_forwarding_proof``. Enforces algorithm pinning (a registered-key
    signer MUST be PS256; a strict node rejects HMAC), non-empty ``forwarded_to``
    bound to this node, ``from_node == forwarded_by``, expiry, and single-use
    nonces."""

    def __init__(self, node_id: str, hmac_secret: str, strict_asymmetric: bool = False) -> None:
        self.node_id = node_id
        self.hmac_secret = hmac_secret
        self.strict_asymmetric = strict_asymmetric
        self.peer_public_keys: dict[str, Any] = {}   # node_id -> PEM (str/bytes)
        self._seen_nonces: dict[str, float] = {}
        self._max_nonces = 100_000

    def register_peer_public_key(self, node_id: str, pem) -> None:
        self.peer_public_keys[node_id] = pem

    def verify(self, signed: dict, from_node: str, now: Optional[float] = None) -> dict:
        """Verify a signed proof ``{proof, signature, sig_alg}``; return the proof
        dict on success. Consumes the nonce on success (single-use). Raises
        ValueError on any failure."""
        now = time.time() if now is None else now
        proof = signed["proof"]
        signature = signed["signature"]
        sig_alg = signed.get("sig_alg", "HS256")
        signer = proof.get("forwarded_by", "")
        canonical = canonical_proof(proof)

        # Algorithm pinning.
        if self.strict_asymmetric and sig_alg != "PS256":
            raise ValueError("this node requires PS256 proofs")
        if signer in self.peer_public_keys and sig_alg != "PS256":
            raise ValueError(f"signer {signer!r} has a registered key; PS256 required")

        # Signature.
        if sig_alg == "PS256":
            pem = self.peer_public_keys.get(signer)
            if pem is None:
                raise ValueError(f"no registered public key for signer {signer!r}")
            if not verify_ps256_proof(pem, canonical, signature):
                raise ValueError("invalid proof signature")
        else:
            if not hmac_verify_proof(self.hmac_secret, canonical, signature):
                raise ValueError("invalid proof signature")

        # Target binding: non-empty and == self.
        target = proof.get("forwarded_to", "")
        if not target or target != self.node_id:
            raise ValueError(f"proof target {target!r} not this node")
        # Transport sender must match the signature-verified signer.
        if from_node != signer:
            raise ValueError(f"from_node {from_node!r} != forwarded_by {signer!r}")
        # Expiry.
        if float(proof.get("expires_at", 0)) < now:
            raise ValueError("proof expired")
        # Replay: single-use nonce.
        nonce = proof.get("nonce", "")
        if not nonce or nonce in self._seen_nonces:
            raise ValueError("nonce replay or missing")
        if len(self._seen_nonces) >= self._max_nonces:
            self._seen_nonces.clear()
        self._seen_nonces[nonce] = now
        return proof


# ───────────────────────── receiver (server side) ──────────────────────────
class FederationReceiver:
    """Receiver-side federation state for one SMCP connection: holds the ECDH
    session keys established via ``federated_key_exchange`` and the proof
    validator. Mirrors malgra's per-connection ``FedRuntime``/``ConnState``.

    ``key_exchange`` and ``forward`` take the ``parameters`` dict of a
    ``tool_invoke`` for the respective verb and return the tool ``result``.
    """

    def __init__(self, node_id: str, hmac_secret: str, *, issuer_pem=None,
                 strict_asymmetric: bool = False,
                 dispatch: Optional[Callable[[dict, Optional[str], dict], Any]] = None) -> None:
        self.node_id = node_id
        self.issuer_pem = issuer_pem
        self.validator = ProofValidator(node_id, hmac_secret, strict_asymmetric)
        self.sessions: dict[str, bytes] = {}   # from_node -> session key
        # Optional hook to run the forwarded task against this node's real tools.
        # dispatch(task, client_user, forwarding_metadata) -> result (sync). When
        # None, a malgra-style acknowledgement is returned (matches malgra's
        # server, which verifies and acks without running the inner task).
        self._dispatch = dispatch

    def register_peer_public_key(self, node_id: str, pem) -> None:
        self.validator.register_peer_public_key(node_id, pem)

    def key_exchange(self, params: dict) -> dict:
        peer_node = params.get("peer_node")
        peer_pub_hex = params.get("peer_pub_hex")
        if not peer_node or not peer_pub_hex:
            raise ValueError("federated_key_exchange requires peer_node and peer_pub_hex")
        my_pub_hex, key = perform_ecdh_exchange(peer_pub_hex, self.node_id, peer_node)
        self.sessions[peer_node] = key
        return {"peer_pub_hex": my_pub_hex}

    def forward(self, params: dict, now: Optional[float] = None) -> dict:
        from_node = params.get("from_node")
        enc = params.get("encrypted_request")
        if not from_node or not isinstance(enc, dict):
            raise ValueError("federated_forward requires from_node and encrypted_request")
        key = self.sessions.get(from_node)
        if key is None:
            raise ValueError("no ECDH session for sender (federated_key_exchange required first)")

        nonce = bytes.fromhex(enc.get("nonce", ""))
        if len(nonce) != 12:
            raise ValueError("nonce must be 12 bytes")
        # Re-derive and enforce the session-id AAD so ciphertext bound to a
        # different session fails authentication rather than decrypting.
        expected_session_id = session_id_for(self.node_id, from_node)
        claimed = enc.get("session_id", expected_session_id)
        if claimed != expected_session_id:
            raise ValueError("session_id mismatch")
        plaintext = decrypt_session(
            key, nonce, bytes.fromhex(enc.get("encrypted_data", "")),
            bytes.fromhex(enc.get("tag", "")), expected_session_id)
        request = json.loads(plaintext)

        # Forwarding proof: signature/pinning, target + from_node binding, expiry,
        # single-use nonce (the validator's cache persists for this connection).
        proof = self.validator.verify(request["auth_proof"], from_node, now=now)

        # Forwarded client token (RS256), when an issuer key is configured.
        client_user = None
        if self.issuer_pem:
            claims = verify_rs256_token(self.issuer_pem, proof.get("client_jwt", ""))
            client_user = claims.get("user")

        task = request.get("task", {}) if isinstance(request.get("task"), dict) else {}
        metadata = request.get("forwarding_metadata", {})
        if self._dispatch is not None:
            return self._dispatch(task, client_user, metadata)

        now2 = time.time() if now is None else now
        return {
            "status": "success",
            "processed_by": self.node_id,
            "processed_at": float(now2),
            "task_type": task.get("type"),
            "client": client_user,
        }


# ───────────────────────── sender (client side) ────────────────────────────
def build_signed_proof(client_jwt: str, task: dict, target_node: str, forwarder_node: str, *,
                       hmac_secret: Optional[str] = None, private_key_pem=None,
                       now: Optional[float] = None) -> dict:
    """Build + sign a forwarding proof bound to ``target_node``. Uses PS256 when a
    private key is given (unforgeable per-node), else the shared-secret HMAC."""
    now = time.time() if now is None else now
    task_hash = hashlib.sha256(json.dumps(task, sort_keys=True).encode()).hexdigest()
    proof = {
        "client_jwt": client_jwt,
        "forwarded_by": forwarder_node,
        "forwarded_at": now,
        "task_hash": task_hash,
        "forwarded_to": target_node,
        "nonce": str(uuid.uuid4()),
        "expires_at": now + 300,
    }
    canonical = canonical_proof(proof)
    if private_key_pem is not None:
        return {"proof": proof, "signature": sign_ps256_proof(private_key_pem, canonical), "sig_alg": "PS256"}
    if hmac_secret is None:
        raise ValueError("build_signed_proof needs hmac_secret or private_key_pem")
    return {"proof": proof, "signature": hmac_sign_proof(hmac_secret, canonical), "sig_alg": "HS256"}


async def forward_request(invoke: Callable[..., Awaitable[Any]], node_id: str, target_node: str,
                          task: dict, client_jwt: str, *, hmac_secret: Optional[str] = None,
                          private_key_pem=None) -> Any:
    """Sender side of A2A federation over an ``invoke(tool_name, **params)``
    transport (e.g. :meth:`SMCPClient.invoke_tool`). Runs the ECDH key exchange,
    signs a forwarding proof, GCM-encrypts the request, and calls
    ``federated_forward`` on the downstream node. Returns the node's result."""
    ephemeral, my_pub_hex = initiator_ecdh_start()
    ke = await invoke("federated_key_exchange", peer_node=node_id, peer_pub_hex=my_pub_hex)
    peer_pub_hex = ke["peer_pub_hex"]
    key = initiator_ecdh_finish(ephemeral, peer_pub_hex, node_id, target_node)

    signed = build_signed_proof(client_jwt, task, target_node, node_id,
                                hmac_secret=hmac_secret, private_key_pem=private_key_pem)
    payload = {
        "task": task,
        "auth_proof": signed,
        "forwarding_metadata": {
            "original_client": None,
            "forwarding_path": [node_id],
            "task_id": task.get("task_id", str(uuid.uuid4())),
            "timestamp": time.time(),
        },
    }
    enc = encrypt_request(key, session_id_for(node_id, target_node), payload)
    return await invoke("federated_forward", from_node=node_id, encrypted_request=enc)
