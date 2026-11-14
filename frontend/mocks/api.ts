import fixture from "./chat-response.json";
import type { Api } from "../src/api/client";
import { RequestError } from "../src/api/errors";
import type { SessionDetail, SessionSummary } from "../src/types/api";

// Scripted synthetic UI fixture, not retrieval or consultation logic.
// Session storage keeps this tab's demo archive on reload; no active ID is restored.
export function createMockApi(): Api {
  const key = "energytool-synthetic-archive-v1";
  let saved: SessionDetail[] = [];
  try {
    saved = JSON.parse(sessionStorage.getItem(key) || "[]");
  } catch {
    /* empty demo */
  }
  const sessions = new Map(saved.map((item) => [item.session_id, item]));
  const clone = <T>(value: T): T => JSON.parse(JSON.stringify(value));
  const persist = () => {
    try {
      sessionStorage.setItem(key, JSON.stringify([...sessions.values()]));
    } catch {
      /* memory fallback */
    }
  };
  const summary = (s: SessionDetail): SessionSummary => ({
    session_id: s.session_id,
    status: s.status,
    created_at: s.created_at,
    updated_at: s.updated_at,
  });
  const get = (id: string) => {
    const s = sessions.get(id);
    if (!s) throw new RequestError(404);
    return s;
  };
  const active = (id: string) => {
    const s = get(id);
    if (s.status === "closed") throw new RequestError(409);
    return s;
  };
  return {
    async cancelRequest() {
      // Synthetic responses have no server-side pending request to cancel.
    },
    async chat(body, signal) {
      await new Promise<void>((resolve, reject) => {
        if (signal?.aborted) {
          reject(new DOMException("Aborted", "AbortError"));
          return;
        }
        const abort = () => {
          clearTimeout(timer);
          reject(new DOMException("Aborted", "AbortError"));
        };
        const timer = setTimeout(() => {
          signal?.removeEventListener("abort", abort);
          resolve();
        }, 650);
        signal?.addEventListener("abort", abort, { once: true });
      });
      if (!body.message.trim()) throw new RequestError(422);
      const now = new Date().toISOString();
      const s: SessionDetail = body.session_id
        ? active(body.session_id)
        : {
            session_id: crypto.randomUUID(),
            status: "active",
            created_at: now,
            updated_at: now,
            messages: [],
            brief: clone(fixture.brief),
            lead_id: null,
          };
      const first = s.messages.length === 0;
      const answer = first
        ? fixture.answer
        : "Демонстрационный ответ: сообщение принято. Реальные рекомендации и обновление карточки появятся после подключения базы знаний и API. Можно продолжить разговор без контактов или проверить форму заявки.";
      s.messages.push(
        { role: "user", content: body.message, created_at: now },
        { role: "assistant", content: answer, created_at: now },
      );
      s.updated_at = now;
      sessions.set(s.session_id, s);
      persist();
      return {
        ...clone(fixture),
        session_id: s.session_id,
        answer,
        lead_offer: !first && !s.lead_id,
        debug: body.debug ? clone(fixture.debug) : null,
      };
    },
    async close(id) {
      const s = get(id);
      s.status = "closed";
      s.updated_at = new Date().toISOString();
      persist();
      return summary(s);
    },
    async sessions() {
      return [...sessions.values()].map(summary).reverse();
    },
    async session(id) {
      return clone(get(id));
    },
    async lead(body) {
      const s = active(body.session_id);
      if (s.lead_id) throw new RequestError(409);
      if (
        !body.consent ||
        !body.full_name.trim() ||
        !body.company.trim() ||
        !/^[+\d\s()-]{7,25}$/.test(body.phone)
      )
        throw new RequestError(422);
      s.lead_id = crypto.randomUUID();
      s.updated_at = new Date().toISOString();
      persist();
      return {
        lead_id: s.lead_id,
        session_id: s.session_id,
        created_at: s.updated_at,
        summary:
          "Синтетическая заявка сохранена для проверки интерфейса. Объект: склад. Реальные рекомендации не сформированы. Внешней отправки нет.",
      };
    },
  };
}
