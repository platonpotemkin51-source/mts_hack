"""Flat competition entry point: each CSV ID is an independent consultation."""

from __future__ import annotations

import csv
import json
import os
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock
from typing import Callable

os.environ.setdefault("RETRIEVAL_MODE", "hybrid")

from energytool.consultant import Consultant, answer_test_case
from energytool.llm_client import generate
from energytool.retrieval import build_retriever


FALLBACK = "Сейчас не удалось получить достоверный ответ по базе знаний. Пожалуйста, обратитесь к специалисту Energytool."


class RunDiagnostics:
    """Request-safe aggregate counters; never store question or response text."""

    def __init__(self) -> None:
        self._lock = Lock()
        self._counts: Counter[str] = Counter()
        self._largest_token_budget = 0

    def record(self, name: str, value: object | None = None) -> None:
        with self._lock:
            if name in ("prompt_tokens", "completion_tokens", "total_tokens", "context_expansions") and isinstance(value, int):
                self._counts[name] += value
            elif name == "max_tokens_retry" and isinstance(value, int):
                self._counts[name] += 1
                self._largest_token_budget = max(self._largest_token_budget, value)
            elif name == "http_error" and isinstance(value, int):
                self._counts["http_errors"] += 1
                self._counts[f"http_{value}"] += 1
            elif name == "finish_reason" and isinstance(value, str):
                self._counts[f"finish_{value}"] += 1
            else:
                self._counts[name] += 1

    def summary(self, *, workers: int) -> dict:
        names = ("transport_error", "http_errors", "timeout", "empty_answer",
                 "length_truncation", "response_parse_error", "parse_error",
                 "unknown_source", "general_refusal", "safe_unsourced_answer",
                 "auxiliary_error", "emergency_fallback", "deadline")
        mode = os.getenv("RETRIEVAL_MODE", "bm25").strip().lower()
        try:
            configured_tokens: int | str = int(os.getenv("MAX_TOKENS", "").strip() or "2048")
        except ValueError:
            configured_tokens = "invalid"
        return {
            "max_tokens_configured": configured_tokens,
            "max_tokens_largest_retry": self._largest_token_budget or None,
            "max_workers": workers,
            "evidence_completion_configured": os.getenv("EVIDENCE_COMPLETION", "1") != "0",
            "answer_review_configured": os.getenv("ANSWER_REVIEW", "0") == "1",
            "client_facts_configured": os.getenv("CLIENT_FACTS", "1") == "1",
            "answer_completeness_configured": os.getenv("ANSWER_COMPLETENESS", "0") == "1",
            "retrieval_mode": mode if mode in ("bm25", "embeddings", "hybrid") else "unknown",
            "counters": {**{name: self._counts[name] for name in names},
                         **{name: value for name, value in self._counts.items() if name not in names}},
        }


def _questions(path: Path) -> list[tuple[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None or not {"id", "question"}.issubset(reader.fieldnames):
            raise ValueError("questions.csv must contain id and question columns")
        rows = [(str(row["id"] or "").strip(), str(row["question"] or "")) for row in reader]
    ids = [item[0] for item in rows]
    if not ids or any(not item for item in ids) or len(set(ids)) != len(ids):
        raise ValueError("questions.csv must have unique nonempty IDs")
    return rows


def run(*, data_dir: Path | None = None, output: Path | None = None,
        consultant: Consultant | None = None,
        clock: Callable[[], float] = time.monotonic,
        diagnostics: RunDiagnostics | None = None) -> tuple[int, int]:
    """Write an atomic id,answer CSV; return (row count, failed count)."""
    started = clock()
    deadline = started + 19 * 60
    source = Path(data_dir or os.getenv("DATA_DIR", "data"))
    destination = Path(output or "submission.csv")
    rows = _questions(source / "questions.csv")
    diagnostics = diagnostics or RunDiagnostics()
    if consultant is None:
        retriever = build_retriever(source / "knowledge-base.zip",
                                    Path(os.getenv("CACHE_DIR", "runtime/index")))
        consultant = Consultant(
            retriever,
            lambda messages: generate(messages, deadline=deadline,
                                      diagnostic_fn=diagnostics.record),
            competition=True,
            context_expansion=os.getenv("CONTEXT_EXPANSION", "1") != "0",
            diagnostic_fn=diagnostics.record,
        )
    try:
        workers = min(8, max(1, int(os.getenv("MAX_WORKERS", "2"))))
    except ValueError as exc:
        raise ValueError("MAX_WORKERS must be an integer") from exc
    # Keep time for a final model attempt and for writing the complete CSV.
    last_start = deadline - 90
    answers: list[str | None] = [None] * len(rows)
    failed = 0

    def work(index: int, question: str) -> tuple[int, str, bool]:
        if clock() >= last_start:
            diagnostics.record("emergency_fallback")
            return index, FALLBACK, True
        try:
            answer = answer_test_case(question, consultant=consultant).strip()
            if not answer or len(answer) > 10_000:
                raise ValueError("Answer is empty or too long")
            return index, answer, False
        except Exception as exc:
            # Never log proxy response bodies, keys, user content or KB text.
            print(f"Case {index + 1} failed: {type(exc).__name__}", file=sys.stderr)
            diagnostics.record("emergency_fallback")
            return index, FALLBACK, True

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(work, index, question): index
                   for index, (_, question) in enumerate(rows)}
        for future in as_completed(futures):
            index, answer, is_failed = future.result()
            answers[index] = answer
            failed += is_failed
    temporary = destination.with_name(destination.name + ".tmp")
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        with temporary.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(("id", "answer"))
            writer.writerows((item[0], answers[index] or FALLBACK)
                             for index, item in enumerate(rows))
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    return len(rows), failed


def main() -> None:
    diagnostics = RunDiagnostics()
    try:
        count, failed = run(diagnostics=diagnostics)
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"Cannot create submission.csv: {type(exc).__name__}", file=sys.stderr)
        raise SystemExit(1) from exc
    # The platform scores submission.csv. A per-case fallback is a valid CSV
    # answer and must not turn a completed run into a failed submission.
    print(f"submission.csv: {count} rows; fallback answers: {failed}", flush=True)
    try:
        workers = min(8, max(1, int(os.getenv("MAX_WORKERS", "2"))))
    except ValueError:
        workers = 2
    print("diagnostics=" + json.dumps(diagnostics.summary(workers=workers),
                                      ensure_ascii=False, sort_keys=True), file=sys.stderr)


if __name__ == "__main__":
    main()
