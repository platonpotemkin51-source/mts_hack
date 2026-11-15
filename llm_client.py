"""Bounded access to the organizer's OpenAI-compatible chat proxy.

This module has no network side effects on import. The API key is never logged or
included in an exception message.
"""

from __future__ import annotations

import asyncio
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
        _setting("LLM_BASE_URL", "BASE_URL", "OPENAI_BASE_URL")
        and _setting("LLM_API_KEY", "OPENAI_API_KEY")
        and _setting("LLM_MODEL", "OPENAI_MODEL", "GEN_MODEL")
    )


def generate(
    messages: list[dict[str, str]],
    *,
    transport: httpx.BaseTransport | None = None,
    sleep: Callable[[float], None] = time.sleep,
    deadline: float | None = None,
    diagnostic_fn: Callable[[str, object | None], None] | None = None,
) -> str:
    """Generate one answer, retrying only transient failures at most twice."""
    base = _setting("LLM_BASE_URL", "BASE_URL", "OPENAI_BASE_URL").rstrip("/")
    key = _setting("LLM_API_KEY", "OPENAI_API_KEY")
    model = _setting("LLM_MODEL", "OPENAI_MODEL", "GEN_MODEL")
    if not base or not key or not model:
        raise ModelUnavailable("Model proxy is not configured")
    if not base.startswith(("http://", "https://")):
        raise ModelUnavailable("Model proxy URL is invalid")
    try:
        temperature = float(_setting("TEMPERATURE", default="0.2"))
        max_tokens = int(_setting("MAX_TOKENS", default="2048"))
    except ValueError as exc:
        raise ModelUnavailable("Model generation settings are invalid") from exc
    if not 0 <= temperature <= 2 or not 1 <= max_tokens <= 8192:
        raise ModelUnavailable("Model generation settings are out of range")

    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    def event(name: str, value: object | None = None) -> None:
        if diagnostic_fn is not None:
            diagnostic_fn(name, value)

    async def request() -> str:
        try:
            async with httpx.AsyncClient(timeout=25.0, follow_redirects=False,
                                         transport=transport) as client:
                for attempt in range(3):
                    remaining = deadline - time.monotonic() if deadline is not None else 90.0
                    if remaining <= 2.0:
                        event("deadline")
                        raise ModelUnavailable("Model deadline reached")
                    attempt_timeout = min(90.0 if deadline is not None else 25.0, remaining - 2.0)
                    try:
                        response = await asyncio.wait_for(
                            client.post(
                                base + "/chat/completions",
                                headers={"Authorization": "Bearer " + key},
                                json=payload,
                                timeout=httpx.Timeout(attempt_timeout,
                                                      connect=min(8.0, attempt_timeout)),
                            ), timeout=attempt_timeout)
                    except (asyncio.TimeoutError, httpx.TimeoutException) as exc:
                        event("timeout")
                        if attempt == 2:
                            raise ModelUnavailable("Model proxy timed out") from exc
                        sleep(min(0.5 * (attempt + 1), max(0.0, deadline - time.monotonic() - 2.0))
                              if deadline is not None else 0.5 * (attempt + 1))
                        continue
                    except httpx.TransportError as exc:
                        event("transport_error")
                        if attempt == 2:
                            raise ModelUnavailable("Model proxy connection failed") from exc
                        sleep(min(0.5 * (attempt + 1), max(0.0, deadline - time.monotonic() - 2.0))
                              if deadline is not None else 0.5 * (attempt + 1))
                        continue
                    if response.status_code != 200:
                        event("http_error", response.status_code)
                        if response.status_code in (429, 500, 502, 503, 504) and attempt < 2:
                            sleep(min(0.5 * (attempt + 1), max(0.0, deadline - time.monotonic() - 2.0))
                                  if deadline is not None else 0.5 * (attempt + 1))
                            continue
                        if response.status_code in (401, 403):
                            raise ModelUnavailable("Model proxy access was denied")
                        raise ModelUnavailable("Model proxy rejected the request")
                    try:
                        body = response.json()
                        choice = body["choices"][0]
                        content = choice["message"].get("content")
                    except (ValueError, KeyError, IndexError, TypeError, AttributeError) as exc:
                        event("response_parse_error")
                        raise ModelUnavailable("Model proxy returned an invalid response") from exc
                    reason = choice.get("finish_reason")
                    if isinstance(reason, str):
                        event("finish_reason", reason if reason in ("stop", "length", "content_filter", "tool_calls") else "other")
                    usage = body.get("usage")
                    if isinstance(usage, dict):
                        for name in ("prompt_tokens", "completion_tokens", "total_tokens"):
                            count = usage.get(name)
                            if isinstance(count, int) and not isinstance(count, bool) and 0 <= count <= 1_000_000_000:
                                event(name, count)
                    if reason == "length":
                        event("length_truncation")
                        if attempt < 2 and payload["max_tokens"] < 8192:
                            payload["max_tokens"] = min(8192, payload["max_tokens"] * 2)
                            event("max_tokens_retry", payload["max_tokens"])
                            continue
                        raise ModelUnavailable("Model response was truncated")
                    if not isinstance(content, str) or not content.strip():
                        event("empty_answer")
                        if attempt < 2:
                            continue
                        raise ModelUnavailable("Model proxy returned an empty answer")
                    return content.strip()
        except httpx.InvalidURL as exc:
            raise ModelUnavailable("Model proxy URL is invalid") from exc
        raise ModelUnavailable("Model proxy did not respond")

    return asyncio.run(request())
