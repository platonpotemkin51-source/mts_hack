"""Competition answers retain grounding and handoff without a web card."""
import json
import unittest
from consultant import Consultant, _parse_model_answer
from contracts import ChatMessage, ClientBrief, KnowledgeChunk


class Retriever:
    def retrieve(self, query, top_k=6):
        return [KnowledgeChunk('doc', 'invented.md', 'Макет', 'Макет показывает расход.')]


class CompetitionOutputTests(unittest.TestCase):
    def test_answer_only_keeps_sources_without_a_spurious_card_error(self):
        parsed = _parse_model_answer(json.dumps({'answer': 'Макет показывает расход.',
                                     'used_chunk_ids': ['C1'], 'lead_offer': False}), competition=True)
        self.assertNotIn('auxiliary_error', parsed)
        self.assertIsInstance(parsed['brief'], ClientBrief)

    def test_malformed_supplied_card_is_still_diagnosed(self):
        parsed = _parse_model_answer(json.dumps({'answer': 'Ответ.', 'brief': 'broken',
                                     'used_chunk_ids': []}), competition=True)
        self.assertTrue(parsed['auxiliary_error'])

    def test_direct_handoff_keeps_client_need_when_model_forgets_summary(self):
        core = Consultant(Retriever(), lambda _: json.dumps({
            'answer': 'Какие модели приборов?', 'used_chunk_ids': ['C1'], 'lead_offer': False}),
            competition=True)
        result = core.respond(message='Перезвоните +70000000000.',
                              history=[ChatMessage('user', 'Нужен учёт по арендаторам склада с 37 приборами.')],
                              brief=ClientBrief())
        self.assertIn('арендаторам склада с 37 приборами', result.answer)
        self.assertNotIn('Какие модели', result.answer)
        self.assertNotIn('+70000000000', result.brief.manager_summary)

    def test_compact_output_does_not_bypass_source_validation(self):
        result = Consultant(Retriever(), lambda _: json.dumps({
            'answer': 'Бесплатная экономия 91%.', 'used_chunk_ids': ['invented-source']}),
            competition=True).respond(message='Что умеет макет?', history=[], brief=ClientBrief())
        self.assertNotIn('91', result.answer)
        self.assertEqual(result.sources, [])


if __name__ == '__main__':
    unittest.main()
