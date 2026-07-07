"""Deterministic PII tokenization ("camouflage") for memory content.

Detected PII (emails, phone numbers, credit cards, optional name lists and
custom patterns) is replaced with deterministic, reversible tokens *before*
content is indexed, linked, or sent to an embedding provider.  PII therefore
never reaches the search index, the link graph, third-party embedding APIs,
or the persisted store — while determinism keeps memory useful: the same
email produces the same token in every zettel, so semantic search and
auto-linking still correlate entities across memories.

Tokens use AES-SIV (RFC 5297, deterministic authenticated encryption,
misuse-resistant — no nonce management) and look like::

    [pii-email-krvgs43fmnzgk5a...]
    [pii-credit_card-gezdgnbvgy3tq...-4242]     (keep_last4)

A tagged placeholder is deliberately not format-preserving: in a text memory
store nothing requires a token to look like an email, and an explicit
``[pii-...]`` marker tells the consuming LLM the value is redacted.  The
category is bound into the ciphertext as associated data, so a token cannot
be replayed under a different category.

Key material comes from the ``ZETTEL_PII_KEY`` env var (hex or base64;
32, 48, or 64 bytes — AES-SIV uses double-length keys, so 64 bytes gives
AES-256-SIV) or an explicit constructor argument.  Keys are never logged.

Requires the ``cryptography`` package (extra: ``zettelkasten-memory[camouflage]``).
"""

from __future__ import annotations

import base64
import binascii
import logging
import os
import re
from typing import Iterable

logger = logging.getLogger(__name__)

ENV_PII_KEY = "ZETTEL_PII_KEY"

_TOKEN_RE = re.compile(r"\[pii-([a-z0-9_]+)-([a-z2-7]+)(?:-(\d{4}))?\]")

_EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
# 13-19 digits allowing single space/dash separators (validated by Luhn below)
_CARD_RE = re.compile(r"(?<![\d.])(?:\d[ -]?){12,18}\d(?![\d.])")
# international-ish phone: 7+ digits with optional +, spaces, dashes, parens.
# Dots are deliberately excluded from the separator class so IP addresses and
# dotted version strings (e.g. 192.168.1.100, 1.2.3) are never matched.
_PHONE_RE = re.compile(r"(?<![\w.])\+?\d(?:[\d\s()-]{5,18})\d(?![\w.])")
# date shapes (2026-07-06, 12/31/2026) that would otherwise look phone-like
_DATE_RE = re.compile(r"^\d{1,4}[-/]\d{1,2}[-/]\d{1,4}$")


class CamouflageError(Exception):
    """Tokenization or detokenization failed."""


def _luhn_ok(digits: str) -> bool:
    total = 0
    for i, ch in enumerate(reversed(digits)):
        d = int(ch)
        if i % 2 == 1:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return total % 10 == 0


def _decode_key(text: str) -> bytes:
    text = text.strip()
    try:
        raw = bytes.fromhex(text)
        if len(raw) in (32, 48, 64):
            return raw
    except ValueError:
        pass
    for decoder in (base64.b64decode, base64.urlsafe_b64decode):
        try:
            raw = decoder(text)
            if len(raw) in (32, 48, 64):
                return raw
        except (ValueError, binascii.Error):
            continue
    raise CamouflageError("PII key must decode to 32, 48, or 64 bytes (hex or base64)")


class CamouflageCodec:
    """Deterministic, reversible PII tokenizer.

    Args:
        key: 32/48/64-byte key (bytes, or hex/base64 str). Defaults to the
            ``ZETTEL_PII_KEY`` env var.
        categories: which built-in detectors run ("email", "phone",
            "credit_card").
        names: optional explicit words/phrases (e.g. person names) to
            tokenize on exact, case-insensitive match.  Regex-based name
            detection is deliberately not offered.
        extra_patterns: mapping of category name -> regex for custom PII
            (SSN, IBAN, ...).
        keep_last4: append the last four digits to credit-card tokens as a
            human-usable hint. Off by default: those four digits become literal
            token text that is embedded, indexed, and persisted, so enabling it
            deliberately leaks a fragment of the card into the index/store.
        reveal: when False, ``detokenize`` is a no-op and retrieval returns
            tokens instead of plaintext PII.

    Note: built-in detection covers only email, phone, and Luhn-valid card
    numbers. Other PII (SSN, IBAN, IP addresses, dates of birth, passport /
    national-ID numbers, postal addresses) is NOT detected unless supplied via
    ``extra_patterns`` — the "PII never reaches the index/store" guarantee
    holds only for the configured categories.
    """

    def __init__(
        self,
        key: bytes | str | None = None,
        *,
        categories: Iterable[str] = ("email", "phone", "credit_card"),
        names: Iterable[str] | None = None,
        extra_patterns: dict[str, str] | None = None,
        keep_last4: bool = False,
        reveal: bool = True,
    ) -> None:
        try:
            from cryptography.hazmat.primitives.ciphers.aead import AESSIV
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "the 'cryptography' package is required for camouflage. "
                "Install with: pip install 'zettelkasten-memory[camouflage]'"
            ) from exc

        if key is None:
            key = os.environ.get(ENV_PII_KEY)
        if not key:
            raise CamouflageError(f"no PII key: set {ENV_PII_KEY} or pass key= to CamouflageCodec")
        raw = key if isinstance(key, bytes) else _decode_key(key)
        if len(raw) not in (32, 48, 64):
            raise CamouflageError("PII key must be 32, 48, or 64 bytes")
        self._siv = AESSIV(raw)

        self.categories = tuple(categories)
        self.keep_last4 = keep_last4
        self.reveal = reveal

        # Order matters: specific detectors run before the loose phone regex
        # so e.g. a custom SSN pattern wins over generic digit matching.
        self._detectors: list[tuple[str, re.Pattern[str]]] = []
        if "email" in self.categories:
            self._detectors.append(("email", _EMAIL_RE))
        if "credit_card" in self.categories:
            self._detectors.append(("credit_card", _CARD_RE))
        for cat, pattern in (extra_patterns or {}).items():
            if not re.fullmatch(r"[a-z0-9_]+", cat):
                raise CamouflageError(f"category name {cat!r} must be [a-z0-9_]+")
            self._detectors.append((cat, re.compile(pattern)))
        if "phone" in self.categories:
            self._detectors.append(("phone", _PHONE_RE))
        if names:
            escaped = sorted((re.escape(n) for n in names), key=len, reverse=True)
            self._detectors.append(
                ("name", re.compile(r"\b(?:" + "|".join(escaped) + r")\b", re.IGNORECASE))
            )

    def __repr__(self) -> str:  # never expose key material
        cats = ",".join(cat for cat, _ in self._detectors)
        return f"CamouflageCodec(categories=[{cats}], reveal={self.reveal})"

    # ------------------------------------------------------------------

    def _encrypt(self, value: str, category: str) -> str:
        ct = self._siv.encrypt(value.encode("utf-8"), [category.encode("ascii")])
        return base64.b32encode(ct).decode("ascii").lower().rstrip("=")

    def _decrypt(self, token: str, category: str) -> str:
        from cryptography.exceptions import InvalidTag

        padded = token.upper() + "=" * (-len(token) % 8)
        try:
            raw = base64.b32decode(padded)
            return self._siv.decrypt(raw, [category.encode("ascii")]).decode("utf-8")
        except (InvalidTag, ValueError, binascii.Error) as exc:
            raise CamouflageError("detokenization failed: wrong PII key or tampered token") from exc

    @staticmethod
    def _sub_outside_tokens(text: str, pattern: re.Pattern[str], repl) -> str:
        """Apply pattern.sub only to segments that are not already tokens."""
        parts: list[str] = []
        last = 0
        for m in _TOKEN_RE.finditer(text):
            parts.append(pattern.sub(repl, text[last : m.start()]))
            parts.append(m.group(0))
            last = m.end()
        parts.append(pattern.sub(repl, text[last:]))
        return "".join(parts)

    def tokenize(self, text: str) -> str:
        """Replace detected PII in *text* with deterministic tokens."""
        if not text:
            return text

        for category, pattern in self._detectors:

            def _sub(match: re.Match[str], _cat: str = category) -> str:
                value = match.group(0)
                if _cat == "credit_card":
                    digits = re.sub(r"[ -]", "", value)
                    if not (13 <= len(digits) <= 19 and _luhn_ok(digits)):
                        return value  # not a real card number
                    suffix = f"-{digits[-4:]}" if self.keep_last4 else ""
                    return f"[pii-credit_card-{self._encrypt(value, _cat)}{suffix}]"
                if _cat == "phone":
                    if _DATE_RE.match(value.strip()):
                        return value  # a date, not a phone number
                    digits = re.sub(r"\D", "", value)
                    if not (7 <= len(digits) <= 15):  # E.164 caps at 15 digits
                        return value
                return f"[pii-{_cat}-{self._encrypt(value, _cat)}]"

            text = self._sub_outside_tokens(text, pattern, _sub)
        return text

    def detokenize(self, text: str) -> str:
        """Restore original PII values for ``[pii-...]`` tokens in *text*.

        No-op when ``reveal`` is False.  A token that fails authentication
        (wrong key, tampering, or a token-shaped string that was never one of
        ours) is left untouched and a warning is logged — detokenization never
        raises from the read path, so one poisoned memory cannot break
        retrieval for a whole namespace.
        """
        if not text or not self.reveal:
            return text

        def _restore(match: re.Match[str]) -> str:
            category, token = match.group(1), match.group(2)
            try:
                return self._decrypt(token, category)
            except CamouflageError:
                logger.warning("leaving unverifiable pii token in place (category=%s)", category)
                return match.group(0)

        return _TOKEN_RE.sub(_restore, text)
