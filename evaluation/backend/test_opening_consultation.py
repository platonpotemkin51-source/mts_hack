"""Opening a consultation must not become a missing-knowledge refusal."""
import json
import unittest
from consultant import Consultant, _safe_unsourced_answer
from contracts import ChatMessage, ClientBrief, KnowledgeChunk


class Retriever:
    def retrieve(self, query, top_k=6):
        return [KnowledgeChunk('doc', 'invented.md', 'Макет', 'Вымышленный макет хранится в шкафу.')]


class OpeningTests(unittest.TestCase):
    def respond(self, answer, message='Привет, помоги разобраться с расходами на электроэнергию', history=None):
        def model(_):
            return json.dumps({'answer': answer, 'used_chunk_ids': [], 'lead_offer': False,
                               'brief': {'object': None, 'problem': None, 'known_parameters': [],
                                         'recommended_products': [], 'missing_fields': []}})
        return Consultant(Retriever(), model).respond(message=message, history=history or [], brief=ClientBrief())

    def test_exact_reported_clarification_without_question_mark_starts_consultation(self):
        result = self.respond('Уточните, пожалуйста, тип вашего объекта (например, магазин, ТСЖ, производственное предприятие) и количество точек учёта.')
        self.assertTrue(result.answer.startswith('Здравствуйте!'))
        self.assertIn('объект', result.answer)
        self.assertEqual(result.answer.count('?'), 1)
        self.assertEqual(result.sources, [])
        self.assertFalse(result.lead_offer)

    def test_greeting_and_preamble_do_not_discard_the_clarification(self):
        for answer in ('Здравствуйте! Давайте разберёмся. Для начала уточните, пожалуйста, тип объекта.',
                       'Здравствуйте! О каком объекте идёт речь?',
                       'Здравствуйте! Какую задачу вы хотите решить?'):
            with self.subTest(answer=answer):
                result = self.respond(answer)
                self.assertTrue(result.answer.startswith('Здравствуйте!'))
                self.assertEqual(result.answer.count('?'), 1)
                self.assertNotIn('базе', result.answer)

    def test_greeting_and_broad_help_are_not_treated_as_missing_product_facts(self):
        for message in ('Привет!', 'Здравствуйте', 'Нужна консультация по электроэнергии',
                        'Помогите разобраться с расходами', 'Хочу снизить расходы на электричество'):
            with self.subTest(message=message):
                result = self.respond('В предоставленной базе нет сведений.', message=message)
                self.assertTrue(result.answer.startswith('Здравствуйте!'))
                self.assertIn('объект', result.answer)
                self.assertNotIn('базе', result.answer)
                self.assertEqual(result.answer.count('?'), 1)

    def test_already_known_object_is_not_asked_again(self):
        result = self.respond('Уточните тип объекта и количество точек учёта.',
                              message='Привет, у нас склад, нужна консультация.')
        self.assertNotIn('тип объекта', result.answer)
        self.assertEqual(result.answer.count('?'), 1)

    def test_followup_does_not_repeat_the_greeting(self):
        result = self.respond('Уточните количество точек учёта.', message='А что ещё нужно?',
                              history=[ChatMessage('user', 'Привет, у нас склад.')])
        self.assertNotIn('Здравствуйте', result.answer)

    def test_unknown_fact_does_not_become_an_uncited_product_claim(self):
        result = self.respond('Макет гарантированно снижает расходы на 91%.',
                              message='Какую экономию гарантирует макет?')
        self.assertNotIn('91', result.answer)
        self.assertIn('не нашлось', result.answer)

    def test_unsafe_preface_is_not_echoed_with_a_safe_question(self):
        answer = _safe_unsourced_answer('Наша система полностью бесплатная. Уточните тип объекта.')
        self.assertEqual(answer, 'Какой у вас тип объекта?')


if __name__ == '__main__':
    unittest.main()
