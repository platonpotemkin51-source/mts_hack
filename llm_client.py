"""Bounded access to the organizer's OpenAI-compatible chat proxy.

This module has no network side effects on import. The API key is never logged or
included in an exception message.
"""

from __future__ import annotations

import os
import time
from typing import Callable

import httpx


class ModelUnavailable(RuntimeError):
    """The configured model could not provide a usable answer."""


def _setting(*names: str, default: str = "") -> str:
    for name in names:
        value = os.getenv(name, "").strip()
        if value and value != name:
            return value
    return default


def configured() -> bool:
    """Whether the three required proxy settings are present."""
    return bool(
        _setting("LLM_BASE_URL", "OPENAI_BASE_URL")
        and _setting("LLM_API_KEY", "OPENAI_API_KEY")
        and _setting("LLM_MODEL", "OPENAI_MODEL", "GEN_MODEL")
    )


def generate(
    messages: list[dict[str, str]],
    *,
    transport: httpx.BaseTransport | None = None,
    sleep: Callable[[float], None] = time.sleep,
) -> str:
    """Generate one answer, retrying only transient failures at most twice."""
    base = _setting("LLM_BASE_URL", "OPENAI_BASE_URL").rstrip("/")
    key = _setting("LLM_API_KEY", "OPENAI_API_KEY")
    model = _setting("LLM_MODEL", "OPENAI_MODEL", "GEN_MODEL")
    if not base or not key or not model:
        raise ModelUnavailable("Model proxy is not configured")
    if not base.startswith(("http://", "https://")):
        raise ModelUnavailable("Model proxy URL is invalid")
    try:
        temperature = float(_setting("TEMPERATURE", default="0.2"))
        max_tokens = int(_setting("MAX_TOKENS", default="700"))
    except ValueError as exc:
        raise ModelUnavailable("Model generation settings are invalid") from exc
    if not 0 <= temperature <= 2 or not 1 <= max_tokens <= 4096:
        raise ModelUnavailable("Model generation settings are out of range")

    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    try:
        with httpx.Client(timeout=httpx.Timeout(45.0, connect=8.0),
                          follow_redirects=False, transport=transport) as client:
            for attempt in range(3):
                try:
                    response = client.post(
                        base + "/chat/completions",
                        headers={"Authorization": "Bearer " + key},
                        json=payload,
                    )
                except httpx.TransportError as exc:
                    if attempt == 2:
                        raise ModelUnavailable("Model proxy connection failed") from exc
                    sleep(0.5 * (attempt + 1))
                    continue
                if response.status_code in (429, 500, 502, 503, 504):
                    if attempt == 2:
                        raise ModelUnavailable("Model proxy is temporarily unavailable")
                    sleep(0.5 * (attempt + 1))
                    continue
                if response.status_code in (401, 403):
                    raise ModelUnavailable("Model proxy access was denied")
                if response.status_code != 200:
                    raise ModelUnavailable("Model proxy rejected the request")
                try:
                    content = response.json()["choices"][0]["message"]["content"]
                except (ValueError, KeyError, IndexError, TypeError) as exc:
                    raise ModelUnavailable("Model proxy returned an invalid response") from exc
                if not isinstance(content, str) or not content.strip():
                    raise ModelUnavailable("Model proxy returned an empty answer")
                return content.strip()
    except httpx.InvalidURL as exc:
        raise ModelUnavailable("Model proxy URL is invalid") from exc
    raise ModelUnavailable("Model proxy did not respond")
