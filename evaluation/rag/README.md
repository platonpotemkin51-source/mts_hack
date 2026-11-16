# Тесты RAG

`python3 -m unittest discover -s evaluation/rag -p "test_*.py"`

Загрузка и границы ZIP, нарезка, BM25, embeddings через тестовый транспорт,
слияние выдачи и сохранение ограничений. Реального API организаторов локально нет.
`benchmark.py` и `compare.py` оставлены для ручных локальных измерений.
Выбранная конфигурация: [BEST_SUBMISSION](BEST_SUBMISSION.md).
