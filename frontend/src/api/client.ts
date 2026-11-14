import type {
  ChatRequest,
  ChatResponse,
  LeadRequest,
  LeadResponse,
  SessionDetail,
  SessionSummary,
} from "../types/api";
import { RequestError } from "./errors";

export interface Api {
  chat(body: ChatRequest, signal?: AbortSignal): Promise<ChatResponse>;
  cancelRequest(id: string): Promise<void>;
  close(id: string): Promise<SessionSummary>;
  sessions(signal?: AbortSignal): Promise<SessionSummary[]>;
  session(id: string, signal?: AbortSignal): Promise<SessionDetail>;
  lead(body: LeadRequest, signal?: AbortSignal): Promise<LeadResponse>;
}

// Never surface arbitrary server exception text, URLs, or response bodies.
export function friendlyError(error: unknown): string {
  const status = error instanceof RequestError ? error.status : 0;
  if (error instanceof RequestError && error.code === "request_pending")
    return "Ответ ещё формируется. Повторите проверку через несколько секунд.";
  if (error instanceof RequestError && error.code === "request_stale")
    return "В диалоге уже есть новые сообщения. Проверьте архив и начните новый диалог.";
  return (
    (
      {
        404: "Диалог или маршрут недоступен. Начните новый диалог; если ошибка повторится, проверьте подключение API.",
        409: "Диалог закрыт или заявка уже существует. Проверьте архив или начните новый диалог.",
        422: "Проверьте заполнение полей. Сервер не принял данные.",
        503: "Консультант временно недоступен. Переписка остаётся на экране.",
      } as Record<number, string>
    )[status] ??
    "Не удалось получить ответ сервера. Проверьте соединение. Переписка остаётся на экране."
  );
}

export function createHttpApi(
  base = import.meta.env.VITE_API_BASE_URL || "",
): Api {
  async function request<T>(
    path: string,
    body?: unknown,
    signal?: AbortSignal,
    keepalive = false,
  ): Promise<T> {
    const timeout = new AbortController();
    const abort = () => timeout.abort();
    signal?.addEventListener("abort", abort, { once: true });
    if (signal?.aborted) timeout.abort();
    const timer = setTimeout(abort, 90_000);
    try {
      const response = await fetch(`${base.replace(/\/$/, "")}/api${path}`, {
        method: body === undefined ? "GET" : "POST",
        headers:
          body === undefined
            ? undefined
            : { "Content-Type": "application/json" },
        body: body === undefined ? undefined : JSON.stringify(body),
        signal: timeout.signal,
        keepalive,
      });
      if (!response.ok) {
        let code: string | undefined;
        try {
          const body = (await response.json()) as { error?: { code?: string } };
          code = body.error?.code;
        } catch {
          /* Keep only the HTTP status. Never show arbitrary server text. */
        }
        throw new RequestError(response.status, code);
      }
      if (response.status === 204) return undefined as T;
      return (await response.json()) as T;
    } finally {
      clearTimeout(timer);
      signal?.removeEventListener("abort", abort);
    }
  }
  return {
    chat: (body, signal) => request("/chat", body, signal),
    cancelRequest: (id) =>
      request(`/requests/${encodeURIComponent(id)}/close`, {}, undefined, true),
    close: (id) =>
      request(`/sessions/${encodeURIComponent(id)}/close`, {}, undefined, true),
    sessions: (signal) => request("/sessions", undefined, signal),
    session: (id, signal) =>
      request(`/sessions/${encodeURIComponent(id)}`, undefined, signal),
    lead: (body, signal) => request("/leads", body, signal),
  };
}

export const mockEnabled = import.meta.env.VITE_USE_MOCK_API === "true";
export const api: Api = mockEnabled
  ? (await import("../../mocks/api")).createMockApi()
  : createHttpApi();
