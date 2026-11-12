# Источники требований

Сверено 12 ноября 2026 года. Здесь хранятся ссылки и краткие рабочие выводы, а не копии предоставленной базы и тестов.

| Материал | Ссылка |
| --- | --- |
| Кейс МТС Energytool | https://drive.google.com/file/d/1M7r8JTXYr2sbMK8XO6VJFVy9NSwiRE8C/view |
| Папка данных | https://drive.google.com/drive/folders/1EKFXt51aswP5IEAAcIXAcNl_geX1sdW5 |
| Технический регламент | https://docs.google.com/document/d/1wZsqH4xXZXwjiAJCtFQGjTaLxEU3D_tMCSw0yXcbQ0U/edit |
| Разрешённые зависимости | https://drive.google.com/file/d/16lACrDJhq1YHH3NpRALXydnmTHwmGfkV/view |
| Пример сабмита | https://drive.google.com/file/d/1iMSDN115sFiTOtZ3__PD0-bU5OgxSRv6/view |
| База знаний | https://drive.google.com/file/d/18jMUdDN10xJHdmGfjjrKY9HPjd7LsGWG/view |
| Демонстрационные вопросы | https://drive.google.com/file/d/19xVinug6_BI4r6EOqr7dsAjF2LHYVQkf/view |
| Пример выходного CSV | https://drive.google.com/file/d/1hdelMb5KJpXsTgj8tZkk0qf-TrBuIPY9/view |

Ключевые ограничения: корневые Python-модули в сабмите; 20 минут, 8 CPU, 16 ГБ RAM, ZIP до 128 МБ; 5 попыток в день со сбросом в 00:00 МСК; генерация через модель организаторов с именем из среды; непустой ответ для каждого ID. Подробности — `SUBMISSION.md`.

Первый шаг перед реализацией CSV-адаптера — скачать и прочитать пример сабмита и вопросы локально. Каркас ещё не проверял их содержимое и не фиксирует неподтверждённый формат сериализации истории внутри `question`.
