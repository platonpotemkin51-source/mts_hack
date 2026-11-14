import { test, expect } from "@playwright/test";
test("explicit mock fixture, subsequent turns, lead and retained archive", async ({
  page,
}) => {
  await page.goto("/");
  await expect(page.getByText("ДЕМО / MOCK", { exact: false })).toBeVisible();
  await page.getByLabel("Ваше сообщение").fill("Склад");
  await page.getByLabel("Отправить сообщение").click();
  await expect(
    page.getByText("Какую задачу на складе нужно решить в первую очередь?"),
  ).toBeVisible();
  await page.getByLabel("Ваше сообщение").fill("Учёт");
  await page.getByLabel("Отправить сообщение").click();
  await page.getByRole("button", { name: "Оставить заявку" }).click();
  await page.getByLabel("ФИО", { exact: true }).fill("Тестовый Клиент");
  await page.getByLabel("Телефон", { exact: true }).fill("+70000000000");
  await page.getByLabel("Компания", { exact: true }).fill("Демо-компания");
  await page.getByLabel("Согласен сохранить", { exact: false }).check();
  await page
    .getByRole("button", { name: "Сохранить заявку", exact: true })
    .click();
  await expect(page.getByText("Контекст задачи собран")).toBeVisible();
  await page.getByRole("button", { name: "Новый диалог" }).click();
  await page.reload();
  await expect(
    page.getByLabel("Карточка клиента").getByText("Пока не указан"),
  ).toBeVisible();
  await page.getByRole("button", { name: "Архив диалогов" }).click();
  await page.getByRole("button", { name: "Диалог 1", exact: false }).click();
  await expect(
    page.getByText("Заявка сохранена", { exact: true }),
  ).toBeVisible();
  await expect(
    page.getByText("Какую задачу на складе нужно решить в первую очередь?"),
  ).toBeVisible();
});
