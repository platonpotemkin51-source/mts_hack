"""Competition CSV checks with isolated, invented inputs."""

from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from solution import FALLBACK, run


class FakeConsultant:
    def __init__(self):
        self.seen = []

    def respond(self, *, message, history, brief):
        self.seen.append((message, list(history), brief.object))
        if message == "ошибка":
            raise RuntimeError("fake failure")
        return type("Result", (), {"answer": 'Ответ, с "кавычками"\nи новой строкой'})()


class SolutionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.data = self.root / "data"
        self.data.mkdir()

    def write(self, rows):
        with (self.data / "questions.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(("id", "question"))
            writer.writerows(rows)

    def test_independent_history_csv_escaping_and_fallback(self):
        self.write([("a", "Один вопрос"),
                    ("b", "Пользователь: Старый вопрос.\nАссистент: Уточните.\nПользователь: Новый вопрос"),
                    ("c", "ошибка")])
        fake = FakeConsultant()
        output = self.root / "submission.csv"
        count, failed = run(data_dir=self.data, output=output, consultant=fake)
        self.assertEqual((count, failed), (3, 1))
        with output.open(encoding="utf-8", newline="") as stream:
            rows = list(csv.reader(stream))
        self.assertEqual(rows[0], ["id", "answer"])
        self.assertEqual([row[0] for row in rows[1:]], ["a", "b", "c"])
        self.assertEqual(rows[1][1], 'Ответ, с "кавычками"\nи новой строкой')
        self.assertEqual(rows[3][1], FALLBACK)
        self.assertTrue(any(message == "Новый вопрос" and len(history) == 2
                            for message, history, _ in fake.seen))
        self.assertTrue(all(brief is None for _, _, brief in fake.seen))

    def test_duplicate_ids_are_rejected(self):
        self.write([("same", "A"), ("same", "B")])
        with self.assertRaises(ValueError):
            run(data_dir=self.data, output=self.root / "submission.csv", consultant=FakeConsultant())

    def test_runtime_budget_reserves_time_to_write_csv(self):
        self.write([(str(index), "Вымышленный вопрос") for index in range(16)])
        virtual_time = [0.0]
        class SlowConsultant:
            def respond(self, **_kwargs):
                virtual_time[0] += 75.0
                return type("Result", (), {"answer": "Ответ"})()
        with patch.dict("os.environ", {"MAX_WORKERS": "1"}):
            count, failed = run(data_dir=self.data, output=self.root / "budget.csv",
                                consultant=SlowConsultant(), clock=lambda: virtual_time[0])
        self.assertEqual(count, 16)
        self.assertGreater(failed, 0)
        self.assertLess(virtual_time[0], 20 * 60)
        with (self.root / "budget.csv").open(encoding="utf-8", newline="") as stream:
            self.assertEqual(len(list(csv.DictReader(stream))), 16)


if __name__ == "__main__":
    unittest.main()
