"""Service scaffold. Planned chat routes are not implemented yet."""

from fastapi import FastAPI

app = FastAPI(title="Energytool Consultant — scaffold", version="0.1.0")


@app.get("/api/health")
def health() -> dict:
    return {"status": "scaffold", "chat_ready": False}
