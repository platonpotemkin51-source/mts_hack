# Backend — Платон

`app/main.py` реализует health/readiness, чат, сессии, архив и заявки. `app/storage.py` хранит их в SQLite, включая диалоги без заявки. Чат использует общий `consultant.py` и существующий `build_retriever`; модель берётся из переменных окружения. Контракт описан в `../contracts/openapi.json`.

Запуск из корня: `python -m uvicorn backend.app.main:app --reload --port 8000`.

Рекомендуемые области по мере реализации: `app/api/` для HTTP, `app/storage/` для SQLite, `app/services/` для жизненного цикла сессий/заявок. Общее ядро остаётся в корневом `consultant.py`; не переносить его в backend.

Полное задание — `../docs/tasks/PLATON.md`.
