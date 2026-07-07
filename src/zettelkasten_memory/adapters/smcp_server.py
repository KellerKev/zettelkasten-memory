"""SMCP server adapter — serve ZettelMemory tools over the authenticated,
encrypted SMCP v3 channel.

Any SMCP v3 client (terminal agents, remote-execution agents, gateway tunnels)
can consume this server; tools surface client-side as
``smcp__<node>__memory_store`` etc.  Namespace binding is derived from the
authenticated identity: each API key maps to a namespace, the namespace is
embedded in the JWT issued at auth, and every tool call is scoped to it.  A
client-supplied ``namespace`` parameter is ignored.

Run:
    export ZETTEL_SMCP_SECRET_KEY=<shared secret>       # Fernet/HMAC channel
    export ZETTEL_SMCP_API_KEY=<api key>                # single tenant, or:
    export ZETTEL_SMCP_API_KEYS="key1=ns1,key2=ns2"     # multi-tenant map
    python -m zettelkasten_memory.adapters.smcp_server --persist memory.bin --encrypt

All secrets come from environment variables (never CLI flags):
    ZETTEL_SMCP_SECRET_KEY   shared secret for the encrypted channel (required)
    ZETTEL_SMCP_JWT_SECRET   JWT HS256 secret, independent of the channel secret
                             (required for multi-namespace servers; a
                             single-namespace server gets a random per-process
                             secret if unset)
    ZETTEL_SMCP_KDF_SALT     per-deployment KDF salt (must match clients)
    ZETTEL_SMCP_API_KEY      single API key bound to ZETTEL_NAMESPACE/default
    ZETTEL_SMCP_API_KEYS     comma-separated key=namespace pairs
    ZETTEL_SMCP_HOST/PORT    bind address (default 127.0.0.1:8765)
    ZETTEL_SMCP_TOKEN_TTL    JWT lifetime seconds (default 3600)
    ZETTEL_SMCP_MAX_SKEW     accepted message-timestamp skew (default 300, 0=off)
    ZETTEL_SMCP_MAX_MESSAGE_BYTES  websocket frame cap (default 1 MiB)
    (SMCP_* and SCP_* prefixes are accepted as fallbacks for each of these.)

Plus the shared memory settings: ZETTEL_MEMORY_KEY (encryption at rest),
ZETTEL_PII_KEY (camouflage), ZETTEL_NAMESPACE, ZETTEL_MAX_CONTENT_BYTES.

There is no TLS termination here: bind to loopback or a cluster-internal
address, or front with a TLS-terminating proxy/tunnel.  The SMCP envelope
itself encrypts and authenticates every payload.
"""

from __future__ import annotations

import argparse
import asyncio
import hmac as _hmac
import logging
import os
import secrets
import time
from dataclasses import dataclass, field
from typing import Any, Mapping

from zettelkasten_memory.adapters import _tools
from zettelkasten_memory.adapters.smcp_protocol import (
    PROTOCOL_VERSION,
    ReplayGuard,
    SMCPCrypto,
    SMCPProtocolError,
    make_message,
    parse_message,
)
from zettelkasten_memory.core import ZettelMemory

logger = logging.getLogger(__name__)


def _env(name: str, env: Mapping[str, str], default: str = "") -> str:
    """Read config with prefix precedence: ZETTEL_SMCP_* > SMCP_* > SCP_*."""
    for prefix in ("ZETTEL_SMCP_", "SMCP_", "SCP_"):
        value = env.get(prefix + name)
        if value:
            return value
    return default


@dataclass
class SMCPServerConfig:
    host: str = "127.0.0.1"
    port: int = 8765
    node_id: str = "zettel-memory"
    secret_key: str = ""
    jwt_secret: str = ""  # empty -> derived from secret_key
    kdf_salt: str = ""
    api_keys: dict[str, str] = field(default_factory=dict)  # api_key -> namespace
    token_ttl: int = 3600
    max_skew: float = 300.0
    max_message_bytes: int = 1_048_576

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> "SMCPServerConfig":
        env = env if env is not None else os.environ
        secret_key = _env("SECRET_KEY", env)
        if not secret_key:
            raise SystemExit(
                "ZETTEL_SMCP_SECRET_KEY is required (the SMCP channel encrypts "
                "and authenticates every payload with it)"
            )

        api_keys: dict[str, str] = {}
        multi = _env("API_KEYS", env)
        if multi:
            for pair in multi.split(","):
                pair = pair.strip()
                if not pair:
                    continue
                if "=" not in pair:
                    raise SystemExit("ZETTEL_SMCP_API_KEYS entries must be key=namespace pairs")
                key, ns = pair.split("=", 1)
                api_keys[key.strip()] = ns.strip() or "default"
        single = _env("API_KEY", env)
        if single:
            api_keys[single] = env.get(_tools.ENV_NAMESPACE, "default") or "default"
        if not api_keys:
            raise SystemExit(
                "set ZETTEL_SMCP_API_KEY or ZETTEL_SMCP_API_KEYS — the server "
                "refuses to run without client authentication"
            )

        # The JWT signing key must be independent of the channel secret: every
        # client holds the channel secret, so a JWT secret derived from it could
        # be recomputed by any client to forge a token for another namespace.
        # When more than one namespace is served, require an explicit,
        # independent ZETTEL_SMCP_JWT_SECRET so tokens are not cross-forgeable
        # (and so tokens validate across replicas).  A single-namespace server
        # gets a random per-process secret if none is set.
        jwt_secret = _env("JWT_SECRET", env)
        if not jwt_secret and len(set(api_keys.values())) > 1:
            raise SystemExit(
                "ZETTEL_SMCP_JWT_SECRET is required when serving multiple "
                "namespaces: it must be independent of the shared channel "
                "secret, which every client holds and could otherwise use to "
                "forge tokens for another tenant"
            )

        mode = _env("MODE", env)
        if mode and mode.lower() != "simple":
            raise SystemExit(
                f"SMCP mode {mode!r} is not supported yet: only the default "
                "mode (API key -> JWT over the encrypted channel) is implemented"
            )

        return cls(
            host=_env("HOST", env, "127.0.0.1"),
            port=int(_env("PORT", env, "8765")),
            node_id=_env("NODE_ID", env, "zettel-memory"),
            secret_key=secret_key,
            jwt_secret=jwt_secret,
            kdf_salt=_env("KDF_SALT", env),
            api_keys=api_keys,
            token_ttl=int(_env("TOKEN_TTL", env, "3600")),
            max_skew=float(_env("MAX_SKEW", env, "300")),
            max_message_bytes=int(_env("MAX_MESSAGE_BYTES", env, str(1_048_576))),
        )


class ZettelSMCPServer:
    """Serves the six memory tools over SMCP v3."""

    def __init__(
        self,
        memory: ZettelMemory,
        config: SMCPServerConfig,
        persist_path: str | None = None,
        *,
        encrypt: bool | str = "auto",
    ) -> None:
        try:
            import jwt as _jwt  # noqa: F401
            import websockets  # noqa: F401
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "the SMCP server requires websockets, cryptography, and PyJWT. "
                "Install with: pip install 'zettelkasten-memory[smcp]'"
            ) from exc

        self.memory = memory
        self.config = config
        self.persist_path = persist_path
        self.encrypt = encrypt
        self.crypto = SMCPCrypto(config.secret_key, config.kdf_salt)
        # Independent of the channel secret by design (see SMCPServerConfig).
        # An explicit secret is stable across restarts/replicas; otherwise a
        # random per-process secret is used (outstanding tokens are invalidated
        # on restart, and clients simply re-authenticate).
        self._jwt_secret = config.jwt_secret or secrets.token_urlsafe(32)

    # -- message handlers ------------------------------------------------

    def _reply(self, msg_type: str, payload: dict[str, Any]) -> dict[str, Any]:
        return make_message(self.crypto, msg_type, payload)

    def _error(self, request_id: str, error: str) -> dict[str, Any]:
        return self._reply(
            "error", {"request_id": request_id, "error": error, "timestamp": time.time()}
        )

    def _handle_handshake(self, message: dict[str, Any]) -> dict[str, Any]:
        payload = message.get("payload") or {}
        return self._reply(
            "handshake",
            {
                "node_id": self.config.node_id,
                "protocol_version": PROTOCOL_VERSION,
                "capabilities_count": len(_tools.TOOL_SPECS),
                "encryption_enabled": True,
                # mutual auth: echo the client's nonce so it can confirm we
                # hold the shared secret and answered THIS handshake
                "client_nonce": str(payload.get("nonce", "")),
            },
        )

    def _handle_auth(self, message: dict[str, Any]) -> dict[str, Any]:
        import jwt

        api_key = str((message.get("payload") or {}).get("api_key", ""))
        namespace = None
        for configured, ns in self.config.api_keys.items():
            if _hmac.compare_digest(api_key, configured):
                namespace = ns
                break
        if namespace is None:
            logger.warning("auth failed for connection")
            return self._error(message["id"], "Authentication failed")

        now = int(time.time())
        token = jwt.encode(
            {
                "client_id": f"ns:{namespace}",
                "ns": namespace,
                "permissions": ["tool_invoke", "discovery"],
                "iat": now,
                "exp": now + self.config.token_ttl,
            },
            self._jwt_secret,
            algorithm="HS256",
        )
        return self._reply(
            "auth",
            {"status": "success", "token": token, "expires_in": self.config.token_ttl},
        )

    def _authorize(self, payload: dict[str, Any], permission: str) -> dict[str, Any] | None:
        """Verify the JWT (pinned to HS256) and required permission; return claims."""
        import jwt

        token = payload.get("token")
        if not token:
            return None
        try:
            claims = jwt.decode(token, self._jwt_secret, algorithms=["HS256"])
        except jwt.InvalidTokenError:
            return None
        if permission not in claims.get("permissions", []):
            return None
        if not claims.get("ns"):
            return None
        return claims

    def _handle_discovery(self, message: dict[str, Any]) -> dict[str, Any]:
        if self._authorize(message.get("payload") or {}, "discovery") is None:
            return self._error(message["id"], "Unauthorized")
        capabilities = {
            name: {
                "name": name,
                "description": spec["description"],
                "parameters": spec["parameters"],
                "auth_required": True,
            }
            for name, spec in _tools.TOOL_SPECS.items()
        }
        return self._reply("capability_discovery", {"capabilities": capabilities})

    def _handle_tool_invoke(self, message: dict[str, Any]) -> dict[str, Any]:
        payload = message.get("payload") or {}
        claims = self._authorize(payload, "tool_invoke")
        if claims is None:
            return self._error(message["id"], "Unauthorized")
        namespace = claims["ns"]

        tool_name = payload.get("tool_name")
        parameters = dict(payload.get("parameters") or {})
        if "namespace" in parameters:
            logger.warning(
                "client-supplied namespace ignored; connection is bound to %r", namespace
            )
            parameters.pop("namespace")

        if tool_name not in _tools.TOOL_SPECS:
            return self._error(message["id"], f"Tool '{tool_name}' not found")

        try:
            result = self._dispatch(tool_name, parameters, namespace)
        except (ValueError, KeyError, TypeError) as exc:
            return self._error(message["id"], f"Tool execution failed: {exc}")

        return self._reply(
            "tool_response",
            {"tool_name": tool_name, "result": result, "status": "success"},
        )

    def _dispatch(self, tool_name: str, params: dict[str, Any], namespace: str) -> Any:
        mem = self.memory
        if tool_name == "memory_store":
            result = _tools.store(
                mem,
                str(params["content"]),
                params.get("tags"),
                float(params.get("importance", 0.5)),
                params.get("metadata"),
                namespace=namespace,
            )
            self._persist()
            return result
        if tool_name == "memory_search":
            return _tools.search(
                mem, str(params["query"]), int(params.get("limit", 5)), namespace=namespace
            )
        if tool_name == "memory_get":
            return _tools.get(mem, str(params["memory_id"]), namespace=namespace)
        if tool_name == "memory_delete":
            result = _tools.delete(mem, str(params["memory_id"]), namespace=namespace)
            self._persist()
            return result
        if tool_name == "memory_connections":
            return _tools.connections(
                mem, str(params["memory_id"]), int(params.get("depth", 1)), namespace=namespace
            )
        if tool_name == "memory_stats":
            return _tools.stats(mem, namespace=namespace)
        raise KeyError(tool_name)

    def _persist(self) -> None:
        _tools.persist_memory(self.memory, self.persist_path, encrypt=self.encrypt)

    def _handle_message(self, message: dict[str, Any]) -> dict[str, Any]:
        msg_type = message.get("type")
        if msg_type == "handshake":
            return self._handle_handshake(message)
        if msg_type == "auth":
            return self._handle_auth(message)
        if msg_type == "capability_discovery":
            return self._handle_discovery(message)
        if msg_type == "tool_invoke":
            return self._handle_tool_invoke(message)
        if msg_type == "heartbeat":
            return self._reply("heartbeat", {"status": "alive", "timestamp": time.time()})
        return self._error(str(message.get("id", "")), "Unknown message type")

    # -- transport ---------------------------------------------------------

    async def _connection(self, websocket) -> None:
        import json as _json

        import websockets

        peer = getattr(websocket, "remote_address", None)
        logger.info("connection from %s", peer)
        # One replay guard per connection: a signed, still-fresh envelope may
        # not be re-submitted.  Sized to the freshness window so memory stays
        # bounded.  Only meaningful when freshness checking is on.
        replay_guard = ReplayGuard(ttl=self.config.max_skew) if self.config.max_skew else None
        try:
            async for raw in websocket:
                try:
                    message = parse_message(
                        self.crypto,
                        raw,
                        max_skew=self.config.max_skew,
                        replay_guard=replay_guard,
                    )
                    response = self._handle_message(message)
                except SMCPProtocolError as exc:
                    logger.warning("rejected message from %s: %s", peer, exc)
                    await websocket.send(_json.dumps(self._error("", str(exc))))
                    continue
                except Exception as exc:  # never let one bad frame kill the task
                    logger.warning("error handling message from %s: %s", peer, exc)
                    await websocket.send(_json.dumps(self._error("", "internal error")))
                    continue
                await websocket.send(_json.dumps(response))
        except websockets.ConnectionClosed:
            logger.info("connection closed: %s", peer)

    async def serve(self) -> None:
        import websockets

        async with websockets.serve(
            self._connection,
            self.config.host,
            self.config.port,
            max_size=self.config.max_message_bytes,
            ping_interval=20,
            ping_timeout=10,
        ):
            logger.info(
                "SMCP memory server %r listening on ws://%s:%d (%d api key(s))",
                self.config.node_id,
                self.config.host,
                self.config.port,
                len(self.config.api_keys),
            )
            await asyncio.Future()  # run forever


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Zettelkasten Memory SMCP Server (secrets via env vars only)",
    )
    parser.add_argument("--host", type=str, default=None, help="Bind host (default 127.0.0.1)")
    parser.add_argument("--port", type=int, default=None, help="Bind port (default 8765)")
    parser.add_argument("--persist", type=str, default=None, help="Path to store file")
    parser.add_argument("--name", type=str, default=None, help="Node id (default zettel-memory)")
    parser.add_argument("--provider", type=str, default=None, help="Embedding provider")
    parser.add_argument("--model", type=str, default=None, help="Embedding model")
    parser.add_argument("--account", type=str, default=None, help="Snowflake account identifier")
    parser.add_argument("--base-url", type=str, default=None, help="Base URL for ollama/malgra")
    parser.add_argument("--compress", action="store_true", default=False)
    parser.add_argument(
        "--encrypt",
        action="store_true",
        default=False,
        help="Require AES-256-GCM at rest (ZETTEL_MEMORY_* env vars)",
    )
    parser.add_argument(
        "--camouflage",
        action="store_true",
        default=False,
        help="Tokenize PII before indexing/persisting (ZETTEL_PII_KEY env var)",
    )
    parser.add_argument("--no-detokenize", action="store_true", default=False)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")

    config = SMCPServerConfig.from_env()
    if args.host:
        config.host = args.host
    if args.port:
        config.port = args.port
    if args.name:
        config.node_id = args.name

    encrypt: bool | str = "auto"
    if args.encrypt:
        from zettelkasten_memory import crypto

        if not crypto.encryption_available():
            raise SystemExit(
                "--encrypt requires key material: set ZETTEL_MEMORY_KEY, "
                "ZETTEL_MEMORY_KEY_FILE, or ZETTEL_MEMORY_PASSPHRASE"
            )
        encrypt = True

    camouflage = None
    if args.camouflage:
        from zettelkasten_memory.camouflage import CamouflageCodec, CamouflageError

        try:
            camouflage = CamouflageCodec(reveal=not args.no_detokenize)
        except CamouflageError as exc:
            raise SystemExit(str(exc))

    backend = _tools.build_backend(
        provider=args.provider,
        model=args.model,
        account=args.account,
        base_url=args.base_url,
        compression=args.compress,
    )
    memory = _tools.build_memory(args.persist, backend, camouflage=camouflage)
    server = ZettelSMCPServer(memory, config, args.persist, encrypt=encrypt)
    asyncio.run(server.serve())


if __name__ == "__main__":
    main()
