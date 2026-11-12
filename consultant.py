"""Platon owns shared consultation logic for web and independent CSV cases."""

from __future__ import annotations

from contracts import (
    ChatMessage,
    ClientBrief,
    ConsultationResult,
    Retriever,
)


class Consultant:
    def __init__(self, retriever: Retriever) -> None:
        self.retriever = retriever

    def respond(
        self,
        *,
        message: str,
        history: list[ChatMessage],
        brief: ClientBrief,
    ) -> ConsultationResult:
        """History excludes the current message; no hidden session state."""
        raise NotImplementedError("Platon: implement consultation; see docs/tasks/PLATON.md")


def answer_test_case(question: str, *, consultant: Consultant) -> str:
    """Answer one independent input; verify organizer history syntax first."""
    raise NotImplementedError("Platon: implement the independent test-case adapter")
