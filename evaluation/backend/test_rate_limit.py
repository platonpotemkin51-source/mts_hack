"""Honor short provider cooldowns without retrying exhausted daily quotas."""
import time
import unittest
from unittest.mock import patch
import httpx
from energytool.llm_client import generate, ModelUnavailable


class RateLimitTests(unittest.TestCase):
    def run_model(self, advised, deadline=None):
        requests, sleeps = [], []
        def handler(request):
            requests.append(request)
            if len(requests) == 1:
                return httpx.Response(429, headers={'retry-after': advised}, text='private body')
            return httpx.Response(200, json={'choices': [{'message': {'content': 'OK'}}]})
        with patch.dict('os.environ', {'LLM_BASE_URL': 'http://invented.test',
                                     'LLM_API_KEY': 'placeholder', 'LLM_MODEL': 'test'}):
            try:
                result = generate([], transport=httpx.MockTransport(handler), sleep=sleeps.append, deadline=deadline)
            except ModelUnavailable as exc:
                result = exc
        return result, requests, sleeps

    def test_short_cooldown_is_respected_instead_of_immediate_retries(self):
        result, requests, sleeps = self.run_model('12')
        self.assertEqual(result, 'OK')
        self.assertEqual(len(requests), 2)
        self.assertEqual(sleeps, [12.0])

    def test_long_quota_reset_stops_after_one_request_without_leaking_body(self):
        result, requests, sleeps = self.run_model('931')
        self.assertIsInstance(result, ModelUnavailable)
        self.assertNotIn('private', str(result))
        self.assertEqual(len(requests), 1)
        self.assertEqual(sleeps, [])

    def test_cooldown_cannot_exceed_remaining_deadline(self):
        result, requests, sleeps = self.run_model('12', time.monotonic() + 7)
        self.assertIsInstance(result, ModelUnavailable)
        self.assertEqual(len(requests), 1)
        self.assertEqual(sleeps, [])


if __name__ == '__main__':
    unittest.main()
