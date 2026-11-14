import { test, expect, type Page } from "@playwright/test";
import { readFileSync } from "node:fs";
const fixture = JSON.parse(
  readFileSync(new URL("../mocks/chat-response.json", import.meta.url), "utf8"),
);

const blank = {
  object: null,
  problem: null,
  known_parameters: [],
  recommended_products: [],
  missing_fields: [],
};
const now = "2026-11-14T12:00:00Z";
async function backend(
  page: Page,
  options: { delay?: number; status?: number } = {},
) {
  const calls: Array<{
    session_id: string | null;
    message: string;
    debug?: boolean;
  }> = [];
  const leads: unknown[] = [],
    closed: string[] = [];
  await page.route(
    (url) => url.pathname.startsWith("/api/"),
    async (route) => {
      const path = new URL(route.request().url()).pathname;
      if (path === "/api/chat") {
        const body = route.request().postDataJSON();
        calls.push(body);
        if (options.delay)
          await new Promise((resolve) => setTimeout(resolve, options.delay));
        if (options.status)
          return route.fulfill({
            status: options.status,
            json: {
              error: { code: "internal", message: "SECRET /private/trace" },
            },
          });
        return route.fulfill({
          json: {
            ...fixture,
            session_id: body.session_id || `session-${calls.length}`,
            answer: `Ответ ${calls.length}: <script>alert(1)</script>`,
            brief: {
              ...blank,
              object: "Склад",
              known_parameters: [body.message],
            },
            lead_offer: true,
            sources: [
              {
                chunk_id: "synthetic",
                title: "Синтетический источник",
                source: "demo/check.md",
                text: "Собственный тестовый фрагмент.",
                score: 1,
              },
            ],
            debug: body.debug
              ? { retrieval_ms: 12, generation_ms: 31, total_ms: 43 }
              : null,
          },
        });
      }
      if (path.endsWith("/close")) {
        closed.push(path);
        return route.fulfill({
          json: {
            session_id: path.split("/")[3],
            status: "closed",
            created_at: now,
            updated_at: now,
          },
        });
      }
      if (path === "/api/leads") {
        leads.push(route.request().postDataJSON());
        return route.fulfill({
          status: 201,
          json: {
            session_id: "session-1",
            lead_id: "lead-1",
            summary: "Серверное резюме тестовой заявки",
            created_at: now,
          },
        });
      }
      if (path === "/api/sessions")
        return route.fulfill({
          json: [
            {
              session_id: "archived",
              status: "closed",
              created_at: now,
              updated_at: now,
            },
          ],
        });
      return route.fulfill({
        json: {
          session_id: "archived",
          status: "closed",
          created_at: now,
          updated_at: now,
          messages: [
            {
              role: "user",
              content: "Архивное сообщение без заявки",
              created_at: now,
            },
          ],
          brief: blank,
          lead_id: null,
        },
      });
    },
  );
  await page.goto("/");
  return { calls, leads, closed };
}
const send = async (page: Page, text = "Нужен учёт на складе") => {
  await page.getByLabel("Ваше сообщение").fill(text);
  await page.getByLabel("Отправить сообщение").click();
};

test("contract, keyboard, duplicate guard, plain text, debug and server brief", async ({
  page,
}) => {
  const { calls } = await backend(page, { delay: 200 });
  await page.getByLabel("Отладка", { exact: true }).check();
  const input = page.getByLabel("Ваше сообщение");
  await input.fill("Склад");
  await input.press("Shift+Enter");
  await input.press("a");
  expect(await input.inputValue()).toContain("\n");
  await input.press("Enter");
  await input.press("Enter");
  await expect(page.getByText("Ответ 1:", { exact: false })).toBeVisible();
  expect(calls).toHaveLength(1);
  expect(calls[0].session_id).toBeNull();
  expect(calls[0].debug).toBe(true);
  await page.getByText("Источники и время").click();
  await expect(page.getByText("Поиск: 12 мс", { exact: false })).toBeVisible();
  await expect(page.getByText("demo/check.md")).toBeVisible();
  await send(page, "Исправление: офис");
  await expect(page.getByText("Ответ 2:", { exact: false })).toBeVisible();
  expect(calls[1].session_id).toBe("session-1");
  await expect(
    page.getByLabel("Карточка клиента").getByText("Исправление: офис"),
  ).toBeVisible();
});

test("refusal, later consent and server lead summary", async ({ page }) => {
  const { leads } = await backend(page);
  await send(page);
  await page.getByRole("button", { name: "Продолжить без контактов" }).click();
  await send(page, "Продолжим консультацию");
  await expect(page.getByText("Ответ 2:", { exact: false })).toBeVisible();
  await expect(
    page.getByRole("button", { name: "Оставить заявку" }),
  ).toHaveCount(0);
  await page.getByRole("button", { name: "Оформить заявку" }).click();
  await page.getByLabel("ФИО", { exact: true }).fill("Тестовый Клиент");
  await page.getByLabel("Телефон", { exact: true }).fill("+70000000000");
  await page.getByLabel("Компания", { exact: true }).fill("Демо-компания");
  await page
    .getByRole("button", { name: "Сохранить заявку", exact: true })
    .click();
  expect(leads).toHaveLength(0);
  await page.getByLabel("Согласен сохранить", { exact: false }).check();
  await page
    .getByRole("button", { name: "Сохранить заявку", exact: true })
    .dblclick();
  await expect(
    page.getByText("Серверное резюме тестовой заявки"),
  ).toBeVisible();
  expect(leads).toHaveLength(1);
  expect(leads[0]).toMatchObject({ consent: true, session_id: "session-1" });
});

test("new dialogue during generation discards old answer and closes old ID", async ({
  page,
}) => {
  const { calls, closed } = await backend(page, { delay: 700 });
  await send(page);
  await expect(page.getByText("Консультант готовит ответ…")).toBeVisible();
  await page.getByRole("button", { name: "Новый диалог" }).click();
  await expect.poll(() => closed.length).toBe(1);
  expect(closed[0]).toMatch(/^\/api\/requests\/.+\/close$/);
  await expect(page.getByText("Ответ 1:", { exact: false })).toHaveCount(0);
  await expect(
    page.getByLabel("Карточка клиента").getByText("Пока не указан"),
  ).toBeVisible();
  await send(page, "Новый объект");
  await expect(page.getByText("Ответ 2:", { exact: false })).toBeVisible();
  expect(calls[1].session_id).toBeNull();
});

test("close, reopen, archive without lead and reload never resume active context", async ({
  page,
}) => {
  const { calls } = await backend(page);
  await send(page);
  await expect(page.getByText("Ответ 1:", { exact: false })).toBeVisible();
  await page.getByLabel("Закрыть чат", { exact: true }).click();
  await page.getByRole("button", { name: "Открыть чат" }).click();
  await page.getByRole("button", { name: "Архив диалогов" }).click();
  await page.getByRole("button", { name: "Диалог 1", exact: false }).click();
  await expect(page.getByText("Архивное сообщение без заявки")).toBeVisible();
  await expect(page.getByText("Без заявки", { exact: true })).toBeVisible();
  await page
    .getByRole("button", { name: "Консультация", exact: false })
    .click();
  await expect(page.getByText("Архивное сообщение без заявки")).toHaveCount(0);
  await send(page);
  await expect(page.getByText("Ответ 2:", { exact: false })).toBeVisible();
  expect(calls[1].session_id).toBeNull();
  await page.reload();
  await send(page);
  await expect(page.getByText("Ответ 3:", { exact: false })).toBeVisible();
  expect(calls[2].session_id).toBeNull();
});

for (const status of [404, 409, 422, 503])
  test(`safe ${status} error preserves message, no fake response or automatic retry`, async ({
    page,
  }) => {
    const { calls } = await backend(page, { status });
    await send(page);
    await expect(page.getByRole("alert")).toBeVisible();
    await expect(page.getByText("SECRET", { exact: false })).toHaveCount(0);
    await expect(
      page.getByText("Нужен учёт на складе", { exact: true }),
    ).toBeVisible();
    await expect(page.getByLabel("Отправить сообщение")).toBeDisabled();
    expect(calls).toHaveLength(1);
  });

test("offline error is not a synthetic answer", async ({ page }) => {
  await page.route("**/api/chat", (route) => route.abort("failed"));
  await page.goto("/");
  await send(page);
  await expect(page.getByRole("alert")).toContainText(
    "Не удалось получить ответ сервера",
  );
  await expect(page.getByLabel("Отправить сообщение")).toBeDisabled();
});

test("uncertain request retries with the same ID and no duplicate message", async ({ page }) => {
  const calls: Array<{ request_id: string; message: string }> = [];
  await page.route("**/api/chat", (route) => {
    const body = route.request().postDataJSON();
    calls.push(body);
    if (calls.length === 1) {
      return route.fulfill({ status: 503, json: { error: { code: "model_unavailable", message: "internal" } } });
    }
    return route.fulfill({ json: { ...fixture, session_id: "recovered-session", answer: "Восстановленный ответ", brief: blank, lead_offer: false, sources: [], debug: null } });
  });
  await page.goto("/");
  await send(page);
  await expect(page.getByRole("button", { name: "Проверить ответ повторно" })).toBeVisible();
  await page.getByRole("button", { name: "Проверить ответ повторно" }).click();
  await expect(page.getByText("Восстановленный ответ")).toBeVisible();
  expect(calls).toHaveLength(2);
  expect(calls[0].request_id).toBe(calls[1].request_id);
  expect(calls[0].message).toBe(calls[1].message);
  await expect(page.getByText("Нужен учёт на складе", { exact: true })).toHaveCount(1);
});

test("mobile layout, long text and keyboard focus", async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await backend(page);
  await page.getByRole("button", { name: "Ваша задача", exact: true }).click();
  await expect(page.getByLabel("Карточка клиента")).toBeVisible();
  await page.getByRole("button", { name: "Диалог", exact: true }).click();
  await send(page, "ОченьДлинныйТекст".repeat(50));
  await expect(page.getByText("Ответ 1:", { exact: false })).toBeVisible();
  expect(
    await page.evaluate(
      () => document.documentElement.scrollWidth <= innerWidth,
    ),
  ).toBe(true);
  await page.getByLabel("Ваше сообщение").focus();
  await page.keyboard.press("Tab");
  await page.getByRole("button", { name: "Новый диалог" }).click();
  await page.screenshot({
    path: "../evaluation/ui/mobile.png",
    fullPage: true,
  });
});

test("desktop screenshot", async ({ page }) => {
  await page.setViewportSize({ width: 1440, height: 1000 });
  await backend(page);
  await page.screenshot({
    path: "../evaluation/ui/desktop.png",
    fullPage: true,
  });
});
