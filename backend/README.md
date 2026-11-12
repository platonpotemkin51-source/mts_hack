# Backend — Платон

Сейчас `app/main.py` предоставляет только `GET /api/health`. Чат и остальные маршруты описаны в `../contracts/openapi.json`, но ещё не реализованы.

Запуск из корня: `python -m uvicorn backend.app.main:app --reload --port 8000`.

Рекомендуемые области по мере реализации: `app/api/` для HTTP, `app/storage/` для SQLite, `app/services/` для жизненного цикла сессий/заявок. Общее ядро остаётся в корневом `consultant.py`; не переносить его в backend.

Полное задание — `../docs/tasks/PLATON.md`.
