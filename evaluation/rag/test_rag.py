"""Offline contract/security checks using invented documents, never private KB."""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import math
import os
import shutil
import stat
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
from zipfile import ZipFile, ZipInfo

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import chunking
import kb_loader
from chunking import chunk_documents
from contracts import KnowledgeDocument
from kb_loader import KnowledgeBaseError, load_documents
from retrieval import BM25Retriever, build_retriever, tokenize
from evaluation.rag.benchmark import evaluate, matches


def document(text, source="invented.md"):
    return KnowledgeDocument("test", source, "Вымышленный каталог", text)


class RAGTests(unittest.TestCase):
    def setUp(self):
        mode = patch.dict(os.environ, {"RETRIEVAL_MODE": "bm25"})
        mode.start()
        self.addCleanup(mode.stop)
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.kb = self.root / "kb"
        self.kb.mkdir()
        (self.kb / "roof.md").write_text(
            "# Вымышленный каталог\n\n## Кровля\n\n"
            "Кровля нужна для вымышленного макета.\n\n"
            "## Фотография\n\nФотография бумажного самолёта хранится локально.\n",
            encoding="utf-8",
        )

    def zip(self, entries):
        path = self.root / "test.zip"
        with ZipFile(path, "w") as archive:
            for name, body in entries:
                archive.writestr(name, body)
        return path

    def test_zip_directory_equivalence_and_stable_ids(self):
        (self.kb / "nested").mkdir()
        (self.kb / "nested" / "OTHER.MD").write_text("# Ещё\n\nПолный хвост.", encoding="utf-8")
        docs = load_documents(self.kb)
        path = self.zip([(p.relative_to(self.kb).as_posix(), p.read_bytes())
                         for p in reversed([item for item in self.kb.rglob("*")
                                            if item.is_file() and item.suffix.lower() == ".md"])])
        self.assertEqual(docs, load_documents(path))
        self.assertEqual(chunk_documents(docs), chunk_documents(load_documents(path)))
        self.assertTrue(all(not Path(d.source).is_absolute() for d in docs))

    def test_content_change_invalidates_document_id(self):
        before = load_documents(self.kb)[0].document_id
        with (self.kb / "roof.md").open("a", encoding="utf-8") as stream:
            stream.write("\nИзменение.")
        self.assertNotEqual(before, load_documents(self.kb)[0].document_id)

    def test_bom_crlf_and_non_markdown(self):
        path = self.zip([("folder\\doc.MD", b"\xef\xbb\xbf# Title\r\n\r\nTail"), ("image.png", b"\xff")])
        docs = load_documents(path)
        self.assertEqual(len(docs), 1)
        self.assertEqual(docs[0].source, "folder/doc.MD")
        self.assertEqual(docs[0].text, "# Title\n\nTail")

    def test_unsafe_zip_paths_never_extract(self):
        for name in ("../escape.md", "/absolute.md", "C:\\escape.md", "nested/../../escape.md"):
            with self.subTest(name=name), self.assertRaises(KnowledgeBaseError):
                load_documents(self.zip([(name, "# Unsafe\n\nText")]))
        self.assertFalse((self.root / "escape.md").exists())

    def test_normalized_duplicate_paths_are_rejected(self):
        with self.assertRaises(KnowledgeBaseError):
            load_documents(self.zip([("a/b.md", "# A"), ("a\\b.md", "# B")]))

    def test_symlinks_are_rejected(self):
        link = self.root / "linked-kb"
        link.symlink_to(self.kb, target_is_directory=True)
        with self.assertRaises(KnowledgeBaseError):
            load_documents(link)
        (self.kb / "link.md").symlink_to(self.kb / "roof.md")
        with self.assertRaises(KnowledgeBaseError):
            load_documents(self.kb)
        entry = ZipInfo("link.md")
        entry.create_system = 3
        entry.external_attr = (stat.S_IFLNK | 0o777) << 16
        with self.assertRaises(KnowledgeBaseError):
            load_documents(self.zip([(entry, "outside.md")]))

    def test_size_and_entry_limits(self):
        with patch.object(kb_loader, "MAX_DOCUMENT_BYTES", 8):
            with self.assertRaises(KnowledgeBaseError):
                load_documents(self.kb)
            with self.assertRaises(KnowledgeBaseError):
                load_documents(self.zip([("a.md", "x" * 9)]))
        with patch.object(kb_loader, "MAX_TOTAL_BYTES", 12):
            with self.assertRaises(KnowledgeBaseError):
                load_documents(self.zip([("a.md", "abcdefg"), ("b.md", "hijklmn")]))
        with patch.object(kb_loader, "MAX_ENTRIES", 1):
            with self.assertRaises(KnowledgeBaseError):
                load_documents(self.zip([("a.md", "a"), ("b.md", "b")]))

    def test_missing_corrupt_empty_and_invalid_utf8(self):
        with self.assertRaises(FileNotFoundError):
            load_documents(self.root / "missing")
        broken = self.root / "broken.zip"
        broken.write_bytes(b"not a zip")
        with self.assertRaises(KnowledgeBaseError):
            load_documents(broken)
        for entries in ([], [("a.md", b"\xff")], [("a.md", "  ")], [("a.md", "binary\0data")]):
            with self.subTest(entries=entries), self.assertRaises(KnowledgeBaseError):
                load_documents(self.zip(entries))

    def test_sections_keep_tables_conditions_and_tail(self):
        table = "  -------------------\n  Item    Value\n  ------  -----\n  A       37 widgets\n\n  B       82 widgets\n  -------------------"
        condition = "Only for an imaginary test. Never a real product promise."
        text = f"# Вымышленный каталог\n\n## Условия\n\n{table}\n\n{condition}\n\n## Конец\n\nUNIQUE_TAIL"
        with patch.object(chunking, "MAX_SECTION_CHARS", 40):
            chunks = chunk_documents([document(text)])
        self.assertTrue(any(table in c.text and condition in c.text for c in chunks))
        self.assertTrue(any("UNIQUE_TAIL" in c.text for c in chunks))
        self.assertEqual(chunks, chunk_documents([document(text)]))

    def test_heading_ancestry_and_fenced_code(self):
        text = "# Вымышленный каталог\n\n## Родитель\n\n### Дочерний\n\n```md\n# Not a heading\n```\n\nХвост."
        chunks = chunk_documents([document(text)])
        self.assertEqual(len(chunks), 1)
        self.assertIn("Родитель / Дочерний", chunks[0].title)
        self.assertIn("```md\n# Not a heading\n```", chunks[0].text)

    def test_oversized_prose_has_overlap_and_no_missing_paragraphs(self):
        blocks = [f"BLOCK{i} " + "a" * 25 for i in range(12)]
        with patch.object(chunking, "MAX_SECTION_CHARS", 100):
            chunks = chunk_documents([document("\n\n".join(blocks))])
        self.assertGreater(len(chunks), 1)
        for block in blocks:
            self.assertTrue(any(block in c.text for c in chunks))
        for left, right in zip(chunks, chunks[1:]):
            self.assertTrue(any(b in left.text and b in right.text for b in blocks))

    def test_chunk_ids_change_with_content_and_sources(self):
        a = chunk_documents([document("body")])[0]
        b = chunk_documents([document("different")])[0]
        c = chunk_documents([document("body", "other.md")])[0]
        self.assertEqual(len({a.chunk_id, b.chunk_id, c.chunk_id}), 3)

    def test_normalization_units_numbers_and_inflections(self):
        self.assertEqual(tokenize("СЧЁТЧИКИ"), tokenize("счетчика"))
        self.assertEqual(tokenize("крыша"), tokenize("кровля"))
        self.assertEqual(tokenize("фото"), tokenize("фотография"))
        self.assertEqual(tokenize("солнечную"), tokenize("солнечная"))
        self.assertEqual(tokenize("панели"), tokenize("панель"))
        self.assertEqual(tokenize("кабинет"), tokenize("кабинете"))
        self.assertEqual(tokenize("электричестве"), tokenize("электроэнергия"))
        self.assertEqual(tokenize("1 000 000 кВт·ч 0,5%"), tokenize("1000000 kWh 0.5 процент"))
        self.assertNotEqual(tokenize("750 кВт"), tokenize("750 кВт·ч"))
        self.assertNotEqual(tokenize("20 МВт"), tokenize("20 кВт"))

    def test_search_ranking_contract_empty_and_no_match(self):
        retriever = build_retriever(self.kb)
        chunks = retriever.retrieve("крыша", top_k=1)
        self.assertEqual(len(chunks), 1)
        self.assertIn("Кровля", chunks[0].title)
        self.assertTrue(math.isfinite(chunks[0].score) and chunks[0].score > 0)
        for query in ("", "  ", "и в на", "zyxwvuquux", "рецепт борща"):
            self.assertEqual(retriever.retrieve(query), [])
        all_chunks = retriever.retrieve("каталог", top_k=100)
        self.assertEqual(len(all_chunks), len({c.chunk_id for c in all_chunks}))
        self.assertEqual([c.score for c in all_chunks], sorted((c.score for c in all_chunks), reverse=True))

    def test_top_k_and_query_validation(self):
        retriever = build_retriever(self.kb)
        for value in (0, -1, True, 1.5, "2", None):
            with self.subTest(value=value), self.assertRaises(ValueError):
                retriever.retrieve("", value)
        with self.assertRaises(TypeError):
            retriever.retrieve(None)
        with self.assertRaises(ValueError):
            BM25Retriever([])

    def test_cache_reuse_invalidation_and_corruption(self):
        cache = self.root / "cache"
        expected = build_retriever(self.kb, cache).retrieve("кровля")
        with patch("retrieval.chunk_documents", side_effect=AssertionError("cache not reused")):
            self.assertEqual(build_retriever(self.kb, cache).retrieve("кровля"), expected)
        next(cache.glob("*.json")).write_text("{broken", encoding="utf-8")
        with self.assertLogs("retrieval", level="WARNING"):
            self.assertEqual(build_retriever(self.kb, cache).retrieve("кровля"), expected)
        (self.kb / "roof.md").write_text("# Новый\n\nУникальная мозаика.", encoding="utf-8")
        self.assertEqual(build_retriever(self.kb, cache).retrieve("кровля"), [])
        self.assertTrue(build_retriever(self.kb, cache).retrieve("мозаика"))

    def test_unwritable_cache_falls_back_to_memory(self):
        blocked = self.root / "file"
        blocked.write_text("not a directory", encoding="utf-8")
        with self.assertLogs("retrieval", level="WARNING"):
            self.assertTrue(build_retriever(self.kb, blocked / "cache").retrieve("кровля"))

    def test_parallel_queries_do_not_mutate_results(self):
        retriever = build_retriever(self.kb)
        expected = retriever.retrieve("кровля")
        with ThreadPoolExecutor(max_workers=8) as pool:
            results = list(pool.map(retriever.retrieve, ["кровля"] * 100))
        self.assertTrue(all(result == expected for result in results))
        expected.clear()
        self.assertTrue(retriever.retrieve("кровля"))

    def test_flat_import_is_offline_and_has_no_indexing_side_effects(self):
        flat = self.root / "flat"
        flat.mkdir()
        for name in ("retrieval.py", "chunking.py", "kb_loader.py", "contracts.py"):
            shutil.copyfile(ROOT / name, flat / name)
        code = """
import socket
socket.socket = lambda *a, **kw: (_ for _ in ()).throw(AssertionError('network'))
import kb_loader
kb_loader.load_documents = lambda *a: (_ for _ in ()).throw(AssertionError('indexing'))
import retrieval
assert not any(__import__('pathlib').Path('.').glob('runtime*'))
"""
        result = subprocess.run([sys.executable, "-B", "-c", code], cwd=flat,
                                capture_output=True, text=True, timeout=20)
        self.assertEqual(result.returncode, 0, result.stderr)
        result = subprocess.run(
            [sys.executable, "-B", "-c",
             "import socket; socket.socket = lambda *a, **kw: None; "
             "from pathlib import Path; from retrieval import build_retriever; "
             "r = build_retriever(Path(__import__('sys').argv[1])); "
             "assert r.retrieve('кровля')[0].source == 'roof.md'", str(self.kb)],
            cwd=flat, capture_output=True, text=True, timeout=20,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_benchmark_checks_section_and_required_caveat(self):
        retriever = build_retriever(self.kb)
        target = {"source": "roof.md", "title": "Кровля", "contains": ["вымышленного"]}
        chunk = retriever.retrieve("кровля", 1)[0]
        self.assertTrue(matches(chunk, target))
        self.assertFalse(matches(chunk, {**target, "contains": ["MISSING_CONDITION"]}))
        cases = [{"id": "found", "query": "кровля", "targets": [target]},
                 {"id": "missing", "query": "quux", "targets": [target]}]
        metrics = evaluate(retriever, cases, repeats=1)["summary"]
        self.assertEqual(metrics["hit_at_6"], 0.5)
        self.assertEqual(metrics["all_targets_at_6"], 0.5)
        self.assertEqual(metrics["reciprocal_rank"], 0.5)


if __name__ == "__main__":
    unittest.main()
