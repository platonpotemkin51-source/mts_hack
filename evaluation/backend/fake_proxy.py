"""Synthetic local chat proxy for transport tests; never a product model.

Run only in development. The fake response uses invented wording and the IDs of
real retrieved chunks, so it checks plumbing but not answer quality.
"""

from __future__ import annotations

import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


class Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        if self.path != "/chat/completions":
            self.send_error(404)
            return
        try:
            size = int(self.headers.get("Content-Length", "0"))
            if size <= 0 or size > 2_000_000:
                raise ValueError("invalid request size")
            payload = json.loads(self.rfile.read(size))
            supplied = json.loads(payload["messages"][-1]["content"])
            chunks = supplied["knowledge_chunks"]
            message = supplied["current_message"]
            history = supplied["history"]
            old = supplied["current_brief"]
            obj = ("офис" if "офис" in message.lower() else
                   "склад" if "склад" in message.lower() else old.get("object"))
            answer = "Тестовый прокси: найден материал базы. Расскажите подробнее о задаче."
            result = {"answer": answer,
                      "brief": {"object": obj, "problem": "тестовая консультация",
                                "known_parameters": [], "recommended_products": [],
                                "missing_fields": ["цель обращения"]},
                      "lead_offer": len(history) >= 2,
                      "used_chunk_ids": [chunks[0]["chunk_id"]] if chunks else []}
            response = {"choices": [{"message": {"content": json.dumps(result, ensure_ascii=False)}}]}
            body = json.dumps(response, ensure_ascii=False).encode("utf-8")
        except (ValueError, KeyError, TypeError, IndexError):
            self.send_error(422)
            return
        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, _format, *_args):
        # Never log user content, retrieved source text or authorization headers.
        return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--host", default="127.0.0.1")
    args = parser.parse_args()
    print("SYNTHETIC MODEL PROXY: transport tests only", flush=True)
    ThreadingHTTPServer((args.host, args.port), Handler).serve_forever()
