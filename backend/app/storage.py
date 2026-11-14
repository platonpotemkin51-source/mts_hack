"""SQLite session store. Every operation uses a short independent transaction."""

from __future__ import annotations

import hashlib
import json
import sqlite3
import uuid
from contextlib import contextmanager
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

from contracts import ClientBrief, ConsultationResult


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


class StoreError(Exception):
    def __init__(self, code: str, status: int, message: str):
        super().__init__(message)
        self.code, self.status, self.message = code, status, message


class Store:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.initialize()

    @contextmanager
    def connect(self):
        db = sqlite3.connect(self.path, timeout=10.0)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA foreign_keys=ON")
        db.execute("PRAGMA busy_timeout=10000")
        try:
            yield db
            db.commit()
        except BaseException:
            db.rollback()
            raise
        finally:
            db.close()

    def initialize(self) -> None:
        with self.connect() as db:
            db.execute("PRAGMA journal_mode=WAL")
            db.executescript("""
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY, status TEXT NOT NULL,
                    brief_json TEXT NOT NULL, created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL REFERENCES sessions(session_id),
                    request_id TEXT, role TEXT NOT NULL, content TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS messages_by_session
                    ON messages(session_id, id);
                CREATE TABLE IF NOT EXISTS chat_requests (
                    request_id TEXT PRIMARY KEY, session_id TEXT,
                    body_hash TEXT, state TEXT NOT NULL,
                    response_json TEXT, cancel_requested INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS leads (
                    lead_id TEXT PRIMARY KEY, session_id TEXT NOT NULL UNIQUE
                    REFERENCES sessions(session_id), full_name TEXT NOT NULL,
                    phone TEXT NOT NULL, company TEXT NOT NULL,
                    summary TEXT NOT NULL, created_at TEXT NOT NULL
                );
            """)

    def expire_idle(self, minutes: int = 60) -> None:
        cutoff = (datetime.now(timezone.utc) - timedelta(minutes=minutes)).isoformat()
        with self.connect() as db:
            db.execute("UPDATE sessions SET status='closed', updated_at=? "
                       "WHERE status='active' AND updated_at<?", (now(), cutoff))

    def create_session(self) -> dict:
        stamp, session_id = now(), uuid.uuid4().hex
        with self.connect() as db:
            db.execute("INSERT INTO sessions VALUES (?, 'active', ?, ?, ?)",
                       (session_id, json.dumps(asdict(ClientBrief()), ensure_ascii=False), stamp, stamp))
        return {"session_id": session_id, "status": "active", "created_at": stamp, "updated_at": stamp}

    def _session(self, db: sqlite3.Connection, session_id: str) -> sqlite3.Row:
        row = db.execute("SELECT * FROM sessions WHERE session_id=?", (session_id,)).fetchone()
        if row is None:
            raise StoreError("session_not_found", 404, "Диалог не найден.")
        return row

    def begin_chat(self, request_id: str, session_id: str | None, message: str) -> tuple[str, list[dict], ClientBrief, dict | None]:
        digest = hashlib.sha256(((session_id or "") + "\0" + message).encode()).hexdigest()
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            existing = db.execute("SELECT * FROM chat_requests WHERE request_id=?", (request_id,)).fetchone()
            if existing:
                if existing["state"] == "cancelled" or existing["cancel_requested"]:
                    raise StoreError("request_cancelled", 409, "Диалог закрыт.")
                if existing["body_hash"] != digest:
                    raise StoreError("request_conflict", 409, "Идентификатор запроса уже использован.")
                if existing["state"] == "completed":
                    return existing["session_id"], [], ClientBrief(), json.loads(existing["response_json"])
                if existing["state"] == "pending":
                    raise StoreError("request_pending", 409, "Ответ ещё формируется.")
                session_id = existing["session_id"]
                db.execute("UPDATE chat_requests SET state='pending' WHERE request_id=?", (request_id,))
            else:
                if session_id is None:
                    session_id = self.create_session_in(db)
                current = self._session(db, session_id)
                if current["status"] != "active":
                    raise StoreError("session_closed", 409, "Начните новый диалог.")
                db.execute("INSERT INTO chat_requests(request_id,session_id,body_hash,state,created_at) "
                           "VALUES (?,?,?,?,?)", (request_id, session_id, digest, "pending", now()))
                db.execute("INSERT INTO messages(session_id,request_id,role,content,created_at) "
                           "VALUES (?,?,'user',?,?)", (session_id, request_id, message, now()))
            current = self._session(db, session_id)
            if current["status"] != "active":
                raise StoreError("session_closed", 409, "Начните новый диалог.")
            history = [dict(role=row["role"], content=row["content"])
                       for row in db.execute("SELECT role,content FROM messages "
                                             "WHERE session_id=? AND (request_id IS NULL OR request_id<>?) "
                                             "ORDER BY id", (session_id, request_id))]
            brief = ClientBrief(**json.loads(current["brief_json"]))
            return session_id, history, brief, None

    def create_session_in(self, db: sqlite3.Connection) -> str:
        stamp, session_id = now(), uuid.uuid4().hex
        db.execute("INSERT INTO sessions VALUES (?, 'active', ?, ?, ?)",
                   (session_id, json.dumps(asdict(ClientBrief()), ensure_ascii=False), stamp, stamp))
        return session_id

    def complete_chat(self, request_id: str, session_id: str, result: ConsultationResult,
                      debug: bool) -> dict:
        response = {
            "session_id": session_id, "answer": result.answer,
            "sources": [asdict(source) for source in result.sources],
            "brief": asdict(result.brief), "lead_offer": result.lead_offer,
            "debug": asdict(result.timings) if debug else None,
        }
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            current = self._session(db, session_id)
            request = db.execute("SELECT * FROM chat_requests WHERE request_id=?", (request_id,)).fetchone()
            if current["status"] != "active" or request["cancel_requested"]:
                db.execute("UPDATE chat_requests SET state='cancelled' WHERE request_id=?", (request_id,))
                raise StoreError("session_closed", 409, "Начните новый диалог.")
            db.execute("INSERT INTO messages(session_id,request_id,role,content,created_at) "
                       "VALUES (?,?,'assistant',?,?)", (session_id, request_id, result.answer, now()))
            db.execute("UPDATE sessions SET brief_json=?,updated_at=? WHERE session_id=?",
                       (json.dumps(asdict(result.brief), ensure_ascii=False), now(), session_id))
            db.execute("UPDATE chat_requests SET state='completed',response_json=? WHERE request_id=?",
                       (json.dumps(response, ensure_ascii=False), request_id))
        return response

    def fail_chat(self, request_id: str) -> None:
        with self.connect() as db:
            db.execute("UPDATE chat_requests SET state='failed' WHERE request_id=? AND state='pending'",
                       (request_id,))

    def cancel_request(self, request_id: str) -> None:
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT session_id FROM chat_requests WHERE request_id=?", (request_id,)).fetchone()
            if row is None:
                db.execute("INSERT INTO chat_requests(request_id,state,cancel_requested,created_at) "
                           "VALUES (?,'cancelled',1,?)", (request_id, now()))
            else:
                db.execute("UPDATE chat_requests SET cancel_requested=1 WHERE request_id=?", (request_id,))
                if row["session_id"]:
                    db.execute("UPDATE sessions SET status='closed',updated_at=? WHERE session_id=?",
                               (now(), row["session_id"]))

    def close_session(self, session_id: str) -> dict:
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = self._session(db, session_id)
            db.execute("UPDATE sessions SET status='closed',updated_at=? WHERE session_id=?",
                       (now(), session_id))
            fresh = self._session(db, session_id)
            return {key: fresh[key] for key in ("session_id", "status", "created_at", "updated_at")}

    def list_sessions(self) -> list[dict]:
        with self.connect() as db:
            return [dict(row) for row in db.execute(
                "SELECT session_id,status,created_at,updated_at FROM sessions ORDER BY created_at DESC")]

    def session_detail(self, session_id: str) -> dict:
        with self.connect() as db:
            row = self._session(db, session_id)
            lead = db.execute("SELECT lead_id FROM leads WHERE session_id=?", (session_id,)).fetchone()
            messages = [dict(item) for item in db.execute(
                "SELECT role,content,created_at FROM messages WHERE session_id=? ORDER BY id", (session_id,))]
            return {"session_id": row["session_id"], "status": row["status"],
                    "created_at": row["created_at"], "updated_at": row["updated_at"],
                    "messages": messages, "brief": json.loads(row["brief_json"]),
                    "lead_id": lead["lead_id"] if lead else None}

    def create_lead(self, session_id: str, full_name: str, phone: str,
                    company: str) -> dict:
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = self._session(db, session_id)
            if row["status"] != "active":
                raise StoreError("session_closed", 409, "Начните новый диалог.")
            if db.execute("SELECT 1 FROM leads WHERE session_id=?", (session_id,)).fetchone():
                raise StoreError("lead_exists", 409, "Заявка уже сохранена.")
            brief = json.loads(row["brief_json"])
            user_messages = [item[0] for item in db.execute(
                "SELECT content FROM messages WHERE session_id=? AND role='user' ORDER BY id", (session_id,))]
            summary_parts = ["Клиент: " + full_name, "Компания: " + company,
                             "Телефон: " + phone]
            for label, value in (("Объект", brief.get("object")), ("Задача", brief.get("problem"))):
                if value:
                    summary_parts.append(label + ": " + value)
            for label, key in (("Параметры", "known_parameters"),
                               ("Обсуждались решения", "recommended_products"),
                               ("Не уточнено", "missing_fields")):
                if brief.get(key):
                    summary_parts.append(label + ": " + "; ".join(brief[key]))
            if user_messages:
                summary_parts.append("Реплики клиента: " + " | ".join(user_messages[-8:]))
            summary = "\n".join(summary_parts)[:5000]
            lead_id, stamp = uuid.uuid4().hex, now()
            db.execute("INSERT INTO leads VALUES (?,?,?,?,?,?,?)",
                       (lead_id, session_id, full_name, phone, company, summary, stamp))
            return {"lead_id": lead_id, "session_id": session_id,
                    "summary": summary, "created_at": stamp}
