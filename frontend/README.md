# Frontend — Даниил

Здесь предстоит создать Vue 3 + TypeScript + Vite. `package.json` и Vue-компонентов пока нет; команда запуска ещё не доступна.

Существующие файлы:

- `src/types/api.ts` — согласованные TypeScript-типы чата, карточки и заявки.
- `mocks/chat-response.json` — синтетический ответ для независимой разработки.

Не удалять их при инициализации Vite в непустой папке. API описан в `../contracts/openapi.json` и `../docs/CONTRACTS.md`. Полное задание — `../docs/tasks/DANIIL.md`.

Предполагаемые настройки после реализации: `VITE_API_BASE_URL=http://localhost:8000`, `VITE_USE_MOCK_API=true` только при явной разработке на mock. В браузере не должно быть ключей LLM.
