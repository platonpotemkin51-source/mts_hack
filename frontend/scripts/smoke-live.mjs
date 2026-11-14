// Browser smoke against real Docker API + RAG with the synthetic local model.
// It does not validate a real LLM. Run only with LIVE_STACK_SYNTHETIC=1.
import { chromium, expect } from "@playwright/test";

if (process.env.LIVE_STACK_SYNTHETIC !== "1") {
  throw new Error("Set LIVE_STACK_SYNTHETIC=1 to acknowledge the synthetic model");
}
const url = process.env.LIVE_URL || "http://localhost:15173";
const browser = await chromium.launch({ channel: "msedge", headless: true });
try {
  const page = await browser.newPage({ viewport: { width: 1440, height: 1000 } });
  await page.goto(url);
  await expect(page.getByText("ДЕМО / MOCK", { exact: false })).toHaveCount(0);
  await page.getByLabel("Отладка", { exact: true }).check();
  const firstResponse = page.waitForResponse((response) => response.url().endsWith("/api/chat") && response.status() === 200);
  await page.getByLabel("Ваше сообщение").fill("У нас склад, выросли счета за электроэнергию. С чего начать?");
  await page.getByLabel("Отправить сообщение").click();
  const first = await (await firstResponse).json();
  await expect(page.getByText(first.answer, { exact: true })).toBeVisible();
  if (!first.sources.length || !first.debug) throw new Error("Real RAG sources or timings missing");

  const secondResponse = page.waitForResponse((response) => response.url().endsWith("/api/chat") && response.status() === 200);
  await page.getByLabel("Ваше сообщение").fill("Уточню: это офис, а не склад. Нужен учёт.");
  await page.getByLabel("Отправить сообщение").click();
  const second = await (await secondResponse).json();
  if (first.session_id !== second.session_id || second.brief.object !== "офис") {
    throw new Error("Session continuity or brief correction failed");
  }
  await expect(page.getByRole("button", { name: "Оставить заявку" })).toBeVisible();
  await page.getByRole("button", { name: "Оставить заявку" }).click();
  await page.getByLabel("ФИО", { exact: true }).fill("Тестовый Клиент");
  await page.getByLabel("Телефон", { exact: true }).fill("+70000000000");
  await page.getByLabel("Компания", { exact: true }).fill("Демо-компания");
  await page.getByLabel("Согласен сохранить", { exact: false }).check();
  await page.getByRole("button", { name: "Сохранить заявку", exact: true }).click();
  await expect(page.getByText("ЗАЯВКА СОХРАНЕНА")).toBeVisible();
  await page.getByRole("button", { name: "Новый диалог" }).click();
  await page.getByRole("button", { name: "Архив диалогов" }).click();
  await expect(page.getByText("Диалог 1", { exact: false }).first()).toBeVisible();
  await page.setViewportSize({ width: 390, height: 844 });
  if (!(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth))) {
    throw new Error("Mobile viewport overflows horizontally");
  }
  console.log("SYNTHETIC MODEL BROWSER SMOKE OK: real API/RAG, source/timings, correction, lead, archive, mobile");
} finally {
  await browser.close();
}
