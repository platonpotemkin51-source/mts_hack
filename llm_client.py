"""Platon owns proxy configuration, generation and bounded error recovery."""

from __future__ import annotations


def generate(messages: list[dict[str, str]]) -> str:
    """Accept OpenAI-style system/user/assistant messages; never log secrets."""
    raise NotImplementedError("Platon: implement proxy access; see docs/SUBMISSION.md")
