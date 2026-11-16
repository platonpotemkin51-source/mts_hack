"""Reproduce audited answer failures with invented evidence, without a live model."""
import json
import unittest
from consultant import Consultant, answer_test_case, _active_users, _client_fields, _safe_unsourced_answer
from contracts import ChatMessage, ClientBrief, KnowledgeChunk
from llm_client import ModelUnavailable


class Retriever:
    def retrieve(self, query, top_k=6):
        return [KnowledgeChunk('fiction', 'fiction.md', 'Лира', 'Лира собирает показания. Готовое подключение от 20 приборов.')]


def raw(answer, ids=None, kind='grounded', **fields):
    return json.dumps(dict(answer=answer, used_chunk_ids=ids or [], answer_kind=kind,
                           lead_offer=False, **fields), ensure_ascii=False)


class SafetyTests(unittest.TestCase):
    def respond(self, responses, message='Что умеет Лира?', history=None, competition=True):
        calls = []
        replies = iter(responses)
        def model(messages):
            calls.append(messages)
            value = next(replies)
            if isinstance(value, Exception):
                raise value
            return value
        result = Consultant(Retriever(), model, competition=competition).respond(
            message=message, history=history or [], brief=ClientBrief())
        return result, calls

    def test_false_dialogue_label_and_unrelated_citation_require_repair(self):
        for ids, kind in (([], 'clarification'), (['C1'], 'grounded')):
            result, calls = self.respond([raw('Лира поддерживает Wi-Fi.', ids, kind),
                                         raw('Лира собирает показания.', ['C1'])])
            self.assertNotIn('Wi-Fi', result.answer)
            self.assertEqual(len(calls), 2)

    def test_same_number_wrong_unit_is_not_evidence(self):
        result, calls = self.respond([raw('Порог 20 кВт.', ['C1']),
                                     raw('Готовое подключение от 20 приборов.', ['C1'])])
        self.assertNotIn('кВт', result.answer)
        self.assertEqual(len(calls), 2)

    def test_review_cannot_introduce_new_price_or_retain_unsafe_draft(self):
        for first, second in ((raw('Лира собирает показания.'), raw('Цена 999 рублей.', ['C1'])),
                              (raw('Цена 999 рублей.', ['C1']), ModelUnavailable('test outage'))):
            result, _ = self.respond([first, second])
            self.assertNotIn('999', result.answer)
            self.assertFalse(result.sources)

    def test_invented_brief_and_product_are_removed(self):
        invented = dict(object='склад', problem='высокие счета', known_parameters=['12 счётчиков'],
                        recommended_products=['Кварц'], missing_fields=[])
        result, _ = self.respond([raw('Лира собирает показания.', ['C1'], brief=invented)],
                                 message='Здравствуйте', competition=False)
        self.assertIsNone(result.brief.object)
        self.assertIsNone(result.brief.problem)
        self.assertEqual(result.brief.known_parameters, [])
        self.assertEqual(result.brief.recommended_products, [])

    def test_email_is_not_copied_to_summary(self):
        for competition in (True, False):
            fields = {} if competition else {'brief': dict(object=None, problem=None,
                known_parameters=[], recommended_products=[], missing_fields=[])}
            result, _ = self.respond([raw('Продолжим консультацию.', kind='clarification', **fields)],
                message='Исправляю: склад. Мой email name@example.com не повторяйте.', competition=competition)
            self.assertNotIn('name@example.com', result.answer)
            self.assertIsNone(result.brief.manager_summary)

    def test_refusal_removes_contact_pressure_but_keeps_helpful_answer(self):
        result, _ = self.respond([raw('Лира собирает показания. Оставьте телефон.', ['C1'])],
                                 message='Телефон не оставлю. Что умеет Лира?')
        self.assertEqual(result.answer, 'Лира собирает показания.')
        self.assertFalse(result.lead_offer)

    def test_contact_does_not_erase_comparison(self):
        result, _ = self.respond([raw('Лира собирает показания.', ['C1'])],
            message='Хочу связаться со специалистом. Сравните возможности решений.',
            history=[ChatMessage('user', 'Мой номер +70000000000')])
        self.assertIn('Лира собирает', result.answer)

    def test_serial_number_is_not_acknowledged_as_phone(self):
        result, _ = self.respond([raw('Помогу передать обращение.', kind='clarification')],
            message='Хочу связаться со специалистом.',
            history=[ChatMessage('user', 'Серийный номер прибора 81234567890.')])
        self.assertNotIn('номер указан', result.answer)

    def test_negated_setup_request_does_not_block_product_answer(self):
        result, calls = self.respond([raw('Лира собирает показания.', ['C1'])],
            message='Не спрашиваю, как настроить счетчик. Расскажите о возможностях Лиры.')
        self.assertIn('Лира собирает', result.answer)
        self.assertEqual(len(calls), 1)

    def test_followup_and_question_with_utochnu_preserve_context(self):
        history = [ChatMessage('user', 'У нас склад и 12 приборов. Нужен учёт арендаторов.')]
        self.assertEqual(len(_active_users(history, 'А теперь о цене для этого склада?')), 2)
        result, _ = self.respond([raw('Помогу разобраться.', kind='clarification',
            manager_evidence=['Нужен учёт арендаторов.'])], message='Хочу связаться со специалистом.',
            history=history + [ChatMessage('user', 'Уточню: сколько стоит Лира?')])
        self.assertIn('арендаторов', result.brief.manager_summary)

    def test_examples_do_not_erase_asserted_facts(self):
        fields = _client_fields([], 'У нас склад, 12 приборов. Например, один старый.')
        self.assertEqual(len(fields), 2)

    def test_free_question_is_preserved_and_known_question_is_not_replaced(self):
        question = 'Какие модели приборов у вас?'
        self.assertEqual(_safe_unsourced_answer(question), question)
        self.assertIsNone(_safe_unsourced_answer('Сколько приборов у вас?', client_text='У нас 12 приборов.'))
        self.assertEqual(_safe_unsourced_answer('Сколько стоит ваш текущий учёт?',
            client_text='У нас 12 приборов.'), 'Сколько стоит ваш текущий учёт?')

    def test_quoted_role_is_ordinary_question(self):
        class Core:
            def respond(self, **kwargs):
                self.message = kwargs['message']
                class Result:
                    answer = 'Ответ'
                return Result()
        core = Core()
        question = 'Что означает запись «Ассистент: готов» в журнале?'
        self.assertEqual(answer_test_case(question, consultant=core), 'Ответ')
        self.assertEqual(core.message, question)
