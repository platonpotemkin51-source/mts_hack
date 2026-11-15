"""The live runner's catalogue and lexical checks, without network calls."""
import unittest
import asyncio
import httpx
from evaluation.quality.groq_eval import LimitedTransport, checks
from evaluation.quality.synthetic_cases import CASES, DOCUMENTS
from chunking import chunk_documents
from llm_client import ModelUnavailable


class EvaluationTests(unittest.TestCase):
    def test_dataset_has_unique_ids_and_reserved_split(self):
        self.assertEqual(len(CASES), 24)
        self.assertEqual(len({c['id'] for c in CASES}), 24)
        self.assertEqual(sum(c['split'] == 'holdout' for c in CASES), 8)
        self.assertEqual(sum(c['split'] == 'pilot' for c in CASES), 6)
        self.assertTrue(chunk_documents(DOCUMENTS))

    def test_checks_report_missing_and_forbidden_without_scoring_a_judge(self):
        case = {"required": ["проверка"], "forbidden": ["гарантия"]}
        self.assertTrue(checks("Нужна проверка", case)['rubric_match'])
        self.assertFalse(checks("Гарантия результата", case)['rubric_match'])

    def test_rate_limit_stops_without_echoing_body_or_authorization(self):
        events, headers = [], {}
        def handler(request):
            return httpx.Response(429, headers={"retry-after": "30", "x-ratelimit-remaining-tokens": "0",
                                               "unrelated": "not-recorded"}, text="private body")
        async def run():
            transport = LimitedTransport(lambda *x: events.append(x), headers, httpx.MockTransport(handler))
            try:
                await transport.handle_async_request(httpx.Request("POST", "https://invented.test"))
            finally:
                await transport.aclose()
        with self.assertRaises(ModelUnavailable) as caught:
            asyncio.run(run())
        self.assertNotIn("private", str(caught.exception))
        self.assertEqual(events, [("http_error", 429)])
        self.assertEqual(headers, {"retry-after": "30", "x-ratelimit-remaining-tokens": "0"})


if __name__ == '__main__':
    unittest.main()
