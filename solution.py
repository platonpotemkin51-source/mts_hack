"""Flat competition entry point: each CSV ID is an independent consultation."""

from __future__ import annotations

import csv
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Callable

from consultant import Consultant, answer_test_case
from llm_client import generate
from retrieval import build_retriever


FALLBACK = "Сейчас не удалось получить достоверный ответ по базе знаний. Пожалуйста, обратитесь к специалисту Energytool."


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
        clock: Callable[[], float] = time.monotonic) -> tuple[int, int]:
    """Write an atomic id,answer CSV; return (row count, failed count)."""
    started = clock()
    deadline = started + 19 * 60
    source = Path(data_dir or os.getenv("DATA_DIR", "data"))
    destination = Path(output or "submission.csv")
    rows = _questions(source / "questions.csv")
    if consultant is None:
        retriever = build_retriever(source / "knowledge-base.zip",
                                    Path(os.getenv("CACHE_DIR", "runtime/index")))
        consultant = Consultant(retriever, lambda messages: generate(messages, deadline=deadline))
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
            return index, FALLBACK, True
        try:
            answer = answer_test_case(question, consultant=consultant).strip()
            if not answer or len(answer) > 10_000:
                raise ValueError("Answer is empty or too long")
            return index, answer, False
        except Exception as exc:
            # Never log proxy response bodies, keys, user content or KB text.
            print(f"Case {index + 1} failed: {type(exc).__name__}", file=sys.stderr)
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
    try:
        count, failed = run()
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"Cannot create submission.csv: {type(exc).__name__}", file=sys.stderr)
        raise SystemExit(1) from exc
    # The platform scores submission.csv. A per-case fallback is a valid CSV
    # answer and must not turn a completed run into a failed submission.
    print(f"submission.csv: {count} rows; fallback answers: {failed}", flush=True)


if __name__ == "__main__":
    main()
