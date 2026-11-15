"""Bounded live evaluation on invented data only; raw traces stay in runtime/."""
import argparse
from collections import Counter
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
import types
from urllib.parse import urlsplit
import httpx

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from chunking import chunk_documents
from consultant import Consultant, SYSTEM_RULES, COMPLETENESS_RULES, COMPETITION_RULES
from contracts import ChatMessage, ClientBrief
from evaluation.quality.synthetic_cases import CASES, DOCUMENTS
from llm_client import ModelUnavailable, generate
from retrieval import BM25Retriever


CONCISE_RULES = SYSTEM_RULES.split("Верни ТОЛЬКО JSON:")[0] + """
Верни только JSON с answer, used_chunk_ids (короткие ID C1, C2) и lead_offer.
Дай конкретный ответ на последний вопрос, затем максимум одно нужное уточнение.
Обычно до 150 слов. Если есть только уточнение без продуктовых фактов, ID могут
быть пустыми. Не обещай отправку заявки до подтверждения системы.
При переходе к специалисту добавь manager_evidence: до трёх коротких дословных
цитат клиента о текущей задаче, актуальных параметрах и объекте. Не включай
старые исправленные параметры и общие вопросы о других продуктах.
Поле brief для обычного ответа можно не возвращать."""

class LimitedTransport(httpx.AsyncBaseTransport):
    """Record quota headers and stop at the first 429, without retry storms."""
    def __init__(self, events, headers, inner=None):
        self.inner = inner if inner is not None else httpx.AsyncHTTPTransport()
        self.events = events
        self.headers = headers

    async def handle_async_request(self, request):
        response = await self.inner.handle_async_request(request)
        self.headers.update({k: v for k, v in response.headers.items()
                             if k.startswith("x-ratelimit-") or k == "retry-after"})
        if response.status_code == 429:
            self.events("http_error", 429)
            await response.aclose()
            raise ModelUnavailable("Groq rate limit; stopped without retries")
        return response

    async def aclose(self):
        await self.inner.aclose()


def checks(answer, case):
    missing = [pattern for pattern in case["required"] if not re.search(pattern, answer, re.I | re.S)]
    forbidden = [pattern for pattern in case["forbidden"] if re.search(pattern, answer, re.I | re.S)]
    return {"missing_patterns": missing, "forbidden_patterns": forbidden,
            "rubric_match": not missing and not forbidden}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("pilot", "dev", "holdout"), default="pilot")
    parser.add_argument("--variants", nargs="+", choices=("baseline", "review", "concise", "complete", "state", "complete_state", "candidate"), default=["baseline", "candidate"])
    parser.add_argument("--label", required=True)
    parser.add_argument("--max-calls", type=int, default=24)
    parser.add_argument("--replay-from", type=Path, help="Reuse saved baseline raw answers on this exact dataset/model")
    parser.add_argument("--pause", type=float, default=15, help="Pause between actual API calls in seconds (0..60)")
    parser.add_argument("--baseline-ref", default="c00f658", help="Frozen consultant revision for the control")
    parser.add_argument("--ids", nargs="+", help="Optional subset of this split; never changes the frozen cases")
    args = parser.parse_args()
    if not re.fullmatch(r"[a-zA-Z0-9_-]+", args.label) or args.max_calls < 1 or args.variants[0] != "baseline" or not 0 <= args.pause <= 60:
        parser.error("Use a simple label, positive call budget and baseline first")
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env", override=False)
    if urlsplit(os.getenv("LLM_BASE_URL", "")).hostname != "api.groq.com":
        parser.error("This tool is only for the configured Groq endpoint")
    output = ROOT / "runtime/evaluation/groq" / (args.label + ".json")
    if output.exists():
        parser.error("Run label already exists; choose a new label to preserve results")
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest = {"documents": [asdict(d) for d in DOCUMENTS], "cases": CASES}
    digest = hashlib.sha256(json.dumps(manifest, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
    manifest_path = output.parent / "synthetic-v1.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
        parser.error("Frozen dataset changed; create a new dataset version")
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    report = {"commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
              "dataset_sha256": digest, "model": os.getenv("LLM_MODEL"), "split": args.split,
              "temperature": 0, "max_tokens": 2048, "results": [], "calls": 0}
    report["source_sha256"] = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                              for name in ("consultant.py", "retrieval.py", "llm_client.py")}
    saved = {}
    if args.replay_from:
        prior = json.loads(args.replay_from.read_text(encoding="utf-8"))
        if prior["dataset_sha256"] != digest or prior["model"] != report["model"]:
            parser.error("Replay dataset/model mismatch")
        saved = {r["id"]: r["traces"] for r in prior["results"]
                 if r["variant"] == "baseline" and "error" not in r}
    report["replay_from"] = str(args.replay_from) if args.replay_from else None
    report["experimental_prompts"] = {"complete": COMPLETENESS_RULES, "concise": CONCISE_RULES,
                                      "candidate": COMPETITION_RULES}
    os.environ.update(TEMPERATURE="0", MAX_TOKENS="2048", TOP_K="6", EVIDENCE_COMPLETION="1")
    retriever = BM25Retriever(chunk_documents(DOCUMENTS))
    frozen = types.ModuleType("frozen_consultant")
    exec(subprocess.check_output(["git", "show", args.baseline_ref + ":consultant.py"], cwd=ROOT, text=True), frozen.__dict__)
    report["baseline_ref"] = args.baseline_ref
    stopped = None
    selected = [c for c in CASES if c["split"] == args.split and (not args.ids or c["id"] in args.ids)]
    if not selected or (args.ids and {c["id"] for c in selected} != set(args.ids)):
        parser.error("Selected IDs are not in this split")
    for case in selected:
        first_responses = []
        for variant in args.variants:
            os.environ["ANSWER_REVIEW"] = "1" if variant == "review" else "0"
            os.environ["CLIENT_FACTS"] = "1" if variant in ("state", "complete_state", "candidate") else "0"
            os.environ["ANSWER_COMPLETENESS"] = "1" if variant in ("complete", "complete_state") else "0"
            traces, events, consumed, quota, pacing = [], Counter(), 0, {}, 0.0
            def event(name, value=None):
                events[name] += value if name.endswith("tokens") and isinstance(value, int) else 1
                if name == "http_error":
                    events[f"http_{value}"] += 1

            def model(messages):
                nonlocal consumed, pacing
                if variant == "concise":
                    messages = [{**messages[0], "content": CONCISE_RULES}, *messages[1:]]
                from_saved = variant == "baseline" and consumed < len(saved.get(case["id"], []))
                replay = from_saved or (variant == "review" and consumed < len(first_responses))
                if replay:
                    if from_saved:
                        trace = saved[case["id"]][consumed]
                        if trace["messages"] != messages:
                            raise ModelUnavailable("Saved baseline inputs differ; refusing replay")
                        raw = trace["raw"]
                    else:
                        raw = first_responses[consumed]
                else:
                    if report["calls"] >= args.max_calls:
                        raise ModelUnavailable("Local call budget exhausted")
                    report["calls"] += 1
                    raw = generate(messages, deadline=time.monotonic() + 120, diagnostic_fn=event,
                                   transport=LimitedTransport(event, quota))
                    if args.pause:
                        time.sleep(args.pause)
                        pacing += args.pause
                consumed += 1
                traces.append({"messages": messages, "raw": raw, "replayed": replay})
                if variant == "baseline":
                    first_responses.append(raw)
                return raw

            core_class = frozen.Consultant if variant in ("baseline", "review", "concise") else Consultant
            core = core_class(retriever, model, competition=True, context_expansion=True, diagnostic_fn=event)
            start = time.perf_counter()
            try:
                result = core.respond(message=case["message"],
                                      history=[ChatMessage(**x) for x in case["history"]], brief=ClientBrief())
                row = {"id": case["id"], "variant": variant, "answer": result.answer,
                       "brief": asdict(result.brief), "lead_offer": result.lead_offer,
                       **checks(result.answer, case)}
            except ModelUnavailable as exc:
                row = {"id": case["id"], "variant": variant, "error": str(exc)}
            wall = time.perf_counter()-start
            row.update(seconds=round(wall - pacing, 3), pacing_seconds=pacing,
                       traces=traces, events=dict(events), quota=quota)
            report["results"].append(row)
            print(json.dumps({k: v for k, v in row.items() if k not in {"traces", "brief", "answer"}}, ensure_ascii=False), flush=True)
            if any(events.get(f"http_{s}") for s in (401, 403, 429)) or report["calls"] >= args.max_calls:
                stopped = "rate_limit_or_call_budget"
            report["stopped"] = stopped
            output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
            if stopped:
                break
            time.sleep(0.25)
        if stopped:
            break
    print("Saved local traces:", output.relative_to(ROOT), flush=True)


if __name__ == "__main__":
    main()
