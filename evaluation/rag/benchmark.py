"""Repeatable local retrieval evaluation; private details stay under runtime/."""

import argparse
from collections import Counter
import json
import math
from pathlib import Path
import statistics
import sys
import tempfile
import time
import tracemalloc

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from chunking import chunk_documents
from kb_loader import load_documents
from retrieval import BM25Retriever, build_retriever


def matches(chunk, target):
    return (chunk.source == target["source"]
            and target.get("title", "").casefold() in chunk.title.casefold()
            and all(term.casefold() in chunk.text.casefold()
                    for term in target.get("contains", [])))


def evaluate(retriever, cases, repeats=30):
    details = []
    times = []
    for case in cases:
        results = retriever.retrieve(case["query"], top_k=6)
        targets = case["targets"]
        ranks = [next((rank for rank, chunk in enumerate(results, 1)
                       if matches(chunk, target)), None) for target in targets]
        found = [rank for rank in ranks if rank is not None]
        details.append({"id": case["id"], "query": case["query"], "ranks": ranks,
                        "hit_at_1": any(rank == 1 for rank in found),
                        "hit_at_3": any(rank <= 3 for rank in found),
                        "hit_at_6": bool(found),
                        "all_targets_at_6": bool(targets) and len(found) == len(targets),
                        "reciprocal_rank": 1 / min(found) if found else 0,
                        "results": [{"source": c.source, "title": c.title,
                                     "chunk_id": c.chunk_id, "score": c.score} for c in results]})
    for _ in range(repeats):
        for case in cases:
            start = time.perf_counter()
            retriever.retrieve(case["query"], top_k=6)
            times.append((time.perf_counter() - start) * 1000)
    summary = {name: round(statistics.mean(row[name] for row in details), 4)
               for name in ("hit_at_1", "hit_at_3", "hit_at_6", "all_targets_at_6", "reciprocal_rank")}
    summary.update({"cases": len(cases), "search_samples": len(times),
                    "median_ms": round(statistics.median(times), 3),
                    "p95_ms": round(sorted(times)[math.ceil(len(times) * 0.95) - 1], 3)})
    return {"summary": summary, "details": details}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kb", type=Path, default=ROOT / "data/knowledge-base.zip")
    parser.add_argument("--cases", type=Path, required=True, help="Local annotated JSON cases")
    args = parser.parse_args()
    cases = json.loads(args.cases.read_text(encoding="utf-8"))
    if not isinstance(cases, list) or not cases or any(not c.get("targets") for c in cases):
        parser.error("Expected a nonempty list of cases with query and targets")
    documents = load_documents(args.kb)
    chunks = chunk_documents(documents)
    for case in cases:
        for target in case["targets"]:
            if not any(matches(chunk, target) for chunk in chunks):
                raise ValueError(f"Expected target absent from current KB: {case['id']}")
    # Verify full corpus coverage, including every tail and indented table row.
    for document in documents:
        combined = "\n".join(c.text for c in chunks if c.source == document.source)
        if any(line not in combined for line in document.text.splitlines()
               if line.strip() and not line.lstrip().startswith("#")):
            raise AssertionError("Chunking dropped source content")
    report = {"documents": len(documents), "chunks": len(chunks),
              "chunk_characters": {"min": min(len(c.text) for c in chunks),
                                   "median": statistics.median(len(c.text) for c in chunks),
                                   "max": max(len(c.text) for c in chunks)},
              "all_body_lines_preserved": True, "cases_file": str(args.cases),
              "variants": {}}
    for name, expand in (("bm25_normalized", False), ("bm25_aliases", True)):
        start = time.perf_counter()
        retriever = BM25Retriever(chunks, expand=expand)
        index_ms = (time.perf_counter() - start) * 1000
        report["variants"][name] = evaluate(retriever, cases)
        report["variants"][name]["summary"]["postings_build_ms"] = round(index_ms, 2)
    with tempfile.TemporaryDirectory() as cache:
        for label in ("cold_with_cache_write_ms", "warm_with_cache_read_ms"):
            start = time.perf_counter()
            retriever = build_retriever(args.kb, Path(cache))
            report[label] = round((time.perf_counter() - start) * 1000, 2)
    tracemalloc.start()
    measured = build_retriever(args.kb)
    report["python_peak_build_mib"] = round(tracemalloc.get_traced_memory()[1] / 1024**2, 2)
    tracemalloc.stop()
    report["off_topic_returned_counts"] = {
        query: len(measured.retrieve(query))
        for query in ("", "и в на", "рецепт борща", "погода завтра", "Расскажи анекдот про солнечные панели")
    }
    report["document_chunk_counts"] = dict(Counter(c.source for c in chunks))
    output = ROOT / "runtime/evaluation/rag-report.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items()
                      if key not in {"variants", "document_chunk_counts", "cases_file"}}, ensure_ascii=False, indent=2))
    for name, result in report["variants"].items():
        print(name, json.dumps(result["summary"], ensure_ascii=False))
        print("missing_targets:", [d["id"] for d in result["details"] if not d["all_targets_at_6"]])
    print("Private details: runtime/evaluation/rag-report.json")


if __name__ == "__main__":
    main()
