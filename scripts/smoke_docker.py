"""Test the real Compose stack with an invented KB and synthetic model only."""
from __future__ import annotations

import json
import subprocess
from pathlib import Path
from urllib.request import Request, urlopen
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parents[1]
BASE = "http://127.0.0.1:5187"
COMPOSE = ["docker", "compose", "--env-file", "runtime/verification/ports.env",
           "-p", "energytool-check", "-f", "compose.yaml", "-f",
           "evaluation/integration/compose.yaml"]


def compose(*args):
    subprocess.run([*COMPOSE, *args], cwd=ROOT, check=True)


def request(path, body=None):
    data = json.dumps(body).encode() if body is not None else None
    req = Request(BASE + path, data=data, headers={"Content-Type": "application/json"})
    with urlopen(req, timeout=30) as response:
        return json.load(response)


def main():
    folder = ROOT / "runtime/verification"
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "ports.env").write_text("UI_PORT=5187\nAPI_PORT=8187\n")
    with ZipFile(folder / "knowledge-base.zip", "w") as archive:
        archive.writestr("demo.md", "# Синтетический макет\n\n"
                         "Макет помогает обсудить задачу учёта на складе.\n")
    try:
        compose("up", "--build", "-d", "--wait", "--wait-timeout", "180")
        assert request("/api/ready")["chat_ready"]
        first = request("/api/chat", {"message": "Нужен учёт на складе", "debug": True})
        assert first["sources"] and first["debug"]["total_ms"] >= 0
        sid = first["session_id"]
        second = request("/api/chat", {"message": "Уточню: это офис, а не склад",
                                        "session_id": sid})
        assert second["session_id"] == sid and second["brief"]["object"] == "офис"
        lead = request("/api/leads", {"session_id": sid, "full_name": "Тестовый Клиент",
                      "phone": "+70000000000", "company": "Демо", "consent": True})
        assert lead["summary"] and lead["lead_id"]
        request(f"/api/sessions/{sid}/close", {})
        fresh = request("/api/chat", {"message": "Новый объект", "debug": True})
        assert fresh["session_id"] != sid and fresh["brief"]["object"] is None
        compose("restart", "backend")
        compose("up", "-d", "--wait", "--wait-timeout", "180")
        detail = request(f"/api/sessions/{sid}")
        assert detail["status"] == "closed" and detail["lead_id"] == lead["lead_id"]
        assert len(detail["messages"]) == 4
        assert any(s["session_id"] == fresh["session_id"] for s in request("/api/sessions"))
        print("PASS: Docker API/RAG, debug, correction, lead, isolation, persisted archive")
    finally:
        # Only this dedicated test project and its synthetic volume are removed.
        compose("down", "-v")


if __name__ == "__main__":
    main()
