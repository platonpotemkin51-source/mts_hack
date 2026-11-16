"""Run each retrieval mode once per query; save private results for Platon."""

import argparse
import csv
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from evaluation.rag.benchmark import evaluate
from energytool.retrieval import EmbeddingError, build_retriever


def compare(kb, cases, modes, cache_dir):
    report = {}
    previous = os.environ.get("RETRIEVAL_MODE")
    try:
        for mode in modes:
            os.environ["RETRIEVAL_MODE"] = mode
            start = time.perf_counter()
            try:
                retriever = build_retriever(kb, cache_dir)
                build_ms = (time.perf_counter() - start) * 1000
                result = evaluate(retriever, cases, repeats=0)
                report[mode] = {"status": "ok", "build_ms": round(build_ms, 2), **result}
            except (EmbeddingError, ValueError, OSError) as exc:
                report[mode] = {"status": "error", "error": str(exc)}
    finally:
        if previous is None:
            os.environ.pop("RETRIEVAL_MODE", None)
        else:
            os.environ["RETRIEVAL_MODE"] = previous
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kb", type=Path, default=ROOT / "data/knowledge-base.zip")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--questions", type=Path, default=ROOT / "data/questions.csv")
    group.add_argument("--cases", type=Path, help="Optional private JSON annotations, same as benchmark.py")
    parser.add_argument("--modes", nargs="+", choices=("bm25", "embeddings", "hybrid"),
                        default=["bm25", "embeddings", "hybrid"])
    parser.add_argument("--env-file", type=Path, help="Optional local .env; existing environment wins")
    args = parser.parse_args()
    if args.env_file:
        if not args.env_file.is_file():
            parser.error("Environment file not found")
        from dotenv import load_dotenv
        load_dotenv(args.env_file, override=False)
    if args.cases:
        cases = json.loads(args.cases.read_text(encoding="utf-8"))
    else:
        with args.questions.open(encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            if reader.fieldnames != ["id", "question"]:
                parser.error("Expected CSV columns id,question")
            cases = [{"id": row["id"], "query": row["question"]} for row in reader]
    if (not isinstance(cases, list) or not cases
            or any(not isinstance(c, dict) or not isinstance(c.get("query"), str)
                   or not c["query"].strip() or not c.get("id") for c in cases)):
        parser.error("Expected nonempty queries with IDs")
    if len({c["id"] for c in cases}) != len(cases):
        parser.error("Query IDs must be unique")
    report = compare(args.kb, cases, args.modes, ROOT / "runtime/index")
    output = ROOT / "runtime/evaluation/rag-comparison.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    for mode, result in report.items():
        print(mode, json.dumps({key: value for key, value in result.items() if key != "details"},
                               ensure_ascii=False))
    print("Private queries and source texts: runtime/evaluation/rag-comparison.json")
    return int(any(result["status"] != "ok" for result in report.values()))


if __name__ == "__main__":
    raise SystemExit(main())
