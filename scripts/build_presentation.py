"""Build the public, synthetic-data-only PDF presentation."""

from __future__ import annotations

from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "docs" / "energytool-presentation.pdf"
WIDTH, HEIGHT = 960, 540
RED = colors.HexColor("#E3000B")
INK = colors.HexColor("#18222F")
MUTED = colors.HexColor("#647081")
PAPER = colors.HexColor("#F8F7F4")
LINE = colors.HexColor("#DDE1E5")


def fonts() -> None:
    options = [
        (Path("C:/Windows/Fonts/arial.ttf"), Path("C:/Windows/Fonts/arialbd.ttf")),
        (Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
         Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf")),
    ]
    for regular, bold in options:
        if regular.exists() and bold.exists():
            pdfmetrics.registerFont(TTFont("Energy", str(regular)))
            pdfmetrics.registerFont(TTFont("EnergyBold", str(bold)))
            return
    raise RuntimeError("A Cyrillic TTF font is required to build the presentation")


def text(c: canvas.Canvas, x: int, y: int, value: str, size: int = 18,
         *, bold: bool = False, color=INK) -> None:
    c.setFont("EnergyBold" if bold else "Energy", size)
    c.setFillColor(color)
    c.drawString(x, y, value)


def block(c: canvas.Canvas, x: int, top: int, lines: list[str],
          size: int = 16, leading: int = 25, color=INK) -> None:
    for index, line in enumerate(lines):
        text(c, x, top - index * leading, line, size, color=color)


def card(c: canvas.Canvas, x: int, y: int, w: int, h: int,
         heading: str, lines: list[str], accent=RED) -> None:
    c.setFillColor(colors.white)
    c.roundRect(x, y, w, h, 18, fill=1, stroke=0)
    c.setFillColor(accent)
    c.roundRect(x + 20, y + h - 42, 30, 5, 2, fill=1, stroke=0)
    text(c, x + 20, y + h - 76, heading, 19, bold=True)
    block(c, x + 20, y + h - 110, lines, 13, 21, MUTED)


def base(c: canvas.Canvas, page: int, label: str) -> None:
    c.setFillColor(PAPER)
    c.rect(0, 0, WIDTH, HEIGHT, fill=1, stroke=0)
    c.setFillColor(RED)
    c.rect(0, HEIGHT - 8, WIDTH, 8, fill=1, stroke=0)
    text(c, 48, 32, "МТС Energytool  /  прототип", 10, color=MUTED)
    text(c, 773, 32, f"{label}  {page}/5", 10, color=MUTED)


def arrow(c: canvas.Canvas, x1: int, y: int, x2: int) -> None:
    c.setStrokeColor(RED)
    c.setLineWidth(2)
    c.line(x1, y, x2, y)
    c.line(x2 - 8, y + 5, x2, y)
    c.line(x2 - 8, y - 5, x2, y)


def main() -> None:
    fonts()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    c = canvas.Canvas(str(OUTPUT), pagesize=(WIDTH, HEIGHT), pageCompression=1)
    c.setTitle("МТС Energytool — ИИ-консультант")
    c.setAuthor("Platon Potemkin")

    base(c, 1, "О ПРОЕКТЕ")
    text(c, 48, 432, "Energytool", 55, bold=True)
    text(c, 48, 375, "ИИ-консультант для сайта", 32, bold=True)
    block(c, 48, 318, ["От описания проблемы — к ответу по базе знаний,",
                       "понятной карточке задачи и заявке с резюме."], 20, 31)
    c.setFillColor(colors.white)
    c.roundRect(48, 133, 864, 112, 18, fill=1, stroke=0)
    text(c, 68, 209, "Проверка", 14, bold=True, color=RED)
    text(c, 68, 180, "docker compose up --build", 20, bold=True)
    text(c, 68, 153, "База и параметры прокси задаются локально через data/ и .env", 12, color=MUTED)
    link = "https://github.com/platonpotemkin51-source/mts_hack"
    text(c, 48, 91, link, 13, color=RED)
    c.linkURL(link, (48, 80, 520, 107), relative=0)
    c.showPage()

    base(c, 2, "СЦЕНАРИЙ")
    text(c, 48, 450, "От задачи клиента до заявки", 33, bold=True)
    card(c, 48, 196, 256, 196, "1  Вопрос", ["Клиент описывает объект", "и задачу своими словами.", "Контакты не обязательны."])
    card(c, 352, 196, 256, 196, "2  Консультация", ["Поиск находит источники.", "Модель объясняет решение", "и обновляет карточку."])
    card(c, 656, 196, 256, 196, "3  Заявка", ["Только с согласием:", "ФИО, телефон, компания.", "Архив сохраняет диалог."])
    arrow(c, 309, 291, 348)
    arrow(c, 613, 291, 652)
    text(c, 48, 125, "Новый диалог начинает чистый контекст; прошлые беседы доступны в архиве.", 15)
    c.showPage()

    base(c, 3, "АРХИТЕКТУРА")
    text(c, 48, 450, "Один консультант для веба и CSV", 31, bold=True)
    card(c, 48, 220, 254, 172, "Vue", ["Чат, карточка, архив,", "заявка и отладка."])
    card(c, 353, 220, 254, 172, "FastAPI + SQLite", ["Сессии, идемпотентность,", "история и заявки."])
    card(c, 658, 220, 254, 172, "Ядро", ["Поиск, правила ответа,", "прокси модели."])
    arrow(c, 306, 306, 348)
    arrow(c, 611, 306, 653)
    c.setFillColor(colors.white)
    c.roundRect(48, 100, 864, 76, 16, fill=1, stroke=0)
    text(c, 69, 145, "База знаний", 16, bold=True)
    text(c, 69, 119, "ZIP → фрагменты → BM25 (или embeddings / RRF) → проверяемые источники", 14)
    c.showPage()

    base(c, 4, "ИНТЕРФЕЙС")
    text(c, 48, 450, "Видимый контекст консультации", 32, bold=True)
    screenshot = ROOT / "evaluation" / "ui" / "desktop.png"
    c.drawImage(ImageReader(str(screenshot)), 48, 105, width=570, height=330,
                preserveAspectRatio=True, anchor="c", mask="auto")
    card(c, 651, 190, 261, 208, "Что видно", ["История реплик и карточка", "задачи рядом с чатом.", "Источники и время доступны", "в режиме отладки."])
    block(c, 651, 156, ["Скриншот снят на", "синтетических данных UI."], 12, 18, MUTED)
    c.showPage()

    base(c, 5, "ПРОВЕРКА")
    text(c, 48, 450, "Проверено и что остаётся", 32, bold=True)
    card(c, 48, 225, 410, 167, "Проверено локально", ["RAG, API и браузерные тесты.",
            "Docker: UI + backend + реальный поиск.", "CSV: все 18 демо-ID через тестовый прокси."], accent=colors.HexColor("#17A06B"))
    card(c, 502, 225, 410, 167, "Нужно проверить отдельно", ["Ответы реальной модели Qwen3.8-27B.",
            "Качество embeddings / гибрида.", "Официальный конкурсный прогон."], accent=RED)
    block(c, 48, 178, ["Тестовый прокси проверяет транспорт и интеграцию, а не качество генерации.",
                       "Закрытые материалы и секреты не входят в репозиторий или презентацию."], 14, 25)
    c.showPage()
    c.save()
    print(OUTPUT)


if __name__ == "__main__":
    main()
