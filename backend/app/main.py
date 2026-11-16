"""FastAPI adapter for the shared consultant and persistent session store."""

from __future__ import annotations

import os
import re
import threading
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator
from typing import Literal

from contracts import ChatMessage
from consultant import Consultant, InvalidModelAnswer
from llm_client import ModelUnavailable, configured
from retrieval import build_retriever
from backend.app.storage import Store, StoreError


class ChatRequest(BaseModel):
    session_id: str | None = None
    message: str = Field(min_length=1, max_length=4000)
    debug: bool = False
    request_id: str | None = Field(default=None, min_length=8, max_length=100)

    @field_validator("message")
    @classmethod
    def nonempty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Message cannot be blank")
        return value.strip()


class LeadRequest(BaseModel):
    session_id: str
    full_name: str = Field(min_length=1, max_length=150)
    phone: str = Field(min_length=10, max_length=25)
    company: str = Field(min_length=1, max_length=200)
    consent: Literal[True]

    @field_validator("session_id", "full_name", "phone", "company")
    @classmethod
    def nonempty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Field cannot be blank")
        return value.strip()

    @field_validator("phone")
    @classmethod
    def valid_phone(cls, value: str) -> str:
        if not re.fullmatch(r"\+?[0-9()\-\s]{10,25}", value) or not 10 <= len(re.sub(r"\D", "", value)) <= 15:
            raise ValueError("Invalid phone number")
        return value


def create_app(*, retriever=None, generate_fn=None, db_path: Path | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.store = Store(db_path or Path(os.getenv("SQLITE_PATH", "runtime/energytool.sqlite3")))
        app.state.startup_error = None
        app.state.consultant = None
        try:
            source = retriever or build_retriever(
                Path(os.getenv("DATA_DIR", "data")) / "knowledge-base.zip",
                Path(os.getenv("CACHE_DIR", "runtime/index")),
            )
            options = {"context_expansion": os.getenv("CONTEXT_EXPANSION", "1") != "0"}
            app.state.consultant = (Consultant(source, generate_fn, **options) if generate_fn
                                    else Consultant(source, **options))
        except (OSError, ValueError, RuntimeError) as exc:
            # Health remains available and chat reports 503 until configuration is fixed.
            app.state.startup_error = type(exc).__name__
        yield

    app = FastAPI(title="Energytool Consultant", version="1.0.0", lifespan=lifespan)
    origins = [x.strip() for x in os.getenv("CORS_ORIGINS", "http://localhost:5173").split(",") if x.strip()]
    app.add_middleware(CORSMiddleware, allow_origins=origins, allow_credentials=False,
                       allow_methods=["GET", "POST"], allow_headers=["Content-Type"])
    # Bound memory across long-lived sessions; collisions only serialize requests.
    locks = [threading.Lock() for _ in range(64)]

    def lock_for(key: str) -> threading.Lock:
        return locks[hash(key) % len(locks)]

    def store_for(request: Request) -> Store:
        store = request.app.state.store
        store.expire_idle(int(os.getenv("SESSION_IDLE_MINUTES", "60")))
        return store

    @app.exception_handler(StoreError)
    async def store_error(_request: Request, exc: StoreError):
        return JSONResponse(status_code=exc.status,
                            content={"error": {"code": exc.code, "message": exc.message}})

    @app.exception_handler(RequestValidationError)
    async def validation_error(_request: Request, _exc: RequestValidationError):
        return JSONResponse(status_code=422,
                            content={"error": {"code": "invalid_request", "message": "Проверьте заполнение полей."}})

    @app.get("/api/health")
    def health(request: Request) -> dict:
        ready = bool(request.app.state.consultant) and (bool(generate_fn) or configured())
        return {"status": "ready" if ready else "not_ready", "chat_ready": ready}

    @app.get("/api/ready")
    def ready(request: Request):
        state = health(request)
        if not state["chat_ready"]:
            return JSONResponse(status_code=503, content=state)
        return state

    @app.post("/api/sessions", status_code=201)
    def create_session(request: Request):
        return store_for(request).create_session()

    @app.post("/api/chat")
    def chat(body: ChatRequest, request: Request):
        if not request.app.state.consultant or (not generate_fn and not configured()):
            raise StoreError("service_unavailable", 503, "Консультант временно недоступен.")
        request_id = body.request_id or uuid.uuid4().hex
        key = body.session_id or request_id
        with lock_for(key):
            store = store_for(request)
            session_id, previous, brief, cached = store.begin_chat(request_id, body.session_id, body.message)
            if cached is not None:
                return cached
            try:
                result = request.app.state.consultant.respond(
                    message=body.message,
                    history=[ChatMessage(**item) for item in previous], brief=brief,
                )
                return store.complete_chat(request_id, session_id, result, body.debug)
            except (ModelUnavailable, InvalidModelAnswer) as exc:
                store.fail_chat(request_id)
                raise StoreError("model_unavailable", 503, "Модель временно недоступна.") from exc
            except (OSError, RuntimeError) as exc:
                store.fail_chat(request_id)
                raise StoreError("retrieval_unavailable", 503, "База знаний временно недоступна.") from exc

    @app.post("/api/requests/{request_id}/close", status_code=204)
    def cancel_request(request_id: str, request: Request):
        store_for(request).cancel_request(request_id)

    @app.post("/api/sessions/{session_id}/close")
    def close_session(session_id: str, request: Request):
        return store_for(request).close_session(session_id)

    @app.get("/api/sessions")
    def sessions(request: Request):
        return store_for(request).list_sessions()

    @app.get("/api/sessions/{session_id}")
    def session(session_id: str, request: Request):
        return store_for(request).session_detail(session_id)

    @app.post("/api/leads", status_code=201)
    def lead(body: LeadRequest, request: Request):
        return store_for(request).create_lead(body.session_id, body.full_name,
                                              body.phone, body.company)

    return app


app = create_app()
