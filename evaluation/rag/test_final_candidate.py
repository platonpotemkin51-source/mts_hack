"""Behaviour checks for the isolated candidate built from the scored ZIP."""
import json
import os
import types
import unittest
from unittest.mock import patch
from zipfile import ZipFile

from contracts import ChatMessage, ClientBrief, KnowledgeChunk
from evaluation.rag.build_final_candidate import BASE, candidate_payload


class FixedRetriever:
    def __init__(self, text='Макет собирает показания.'):
        self.text = text

    def retrieve(self, query, top_k=6):
        return [KnowledgeChunk('synthetic', 'fiction.md', 'Макет', self.text)]


@unittest.skipUnless(BASE.exists(), 'Requires the preserved scored ZIP')
class FinalCandidateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.payload = candidate_payload()
        cls.core = types.ModuleType('final_candidate_core')
        exec(cls.payload['consultant.py'], cls.core.__dict__)
        cls.control = types.ModuleType('scored_control_core')
        with ZipFile(BASE) as archive:
            exec(archive.read('consultant.py'), cls.control.__dict__)

    def setUp(self):
        env = patch.dict(os.environ, {'ANSWER_REVIEW': '0', 'CLIENT_FACTS': '1',
                                     'CONTEXT_MODE': 'chunks', 'ANSWER_COMPLETENESS': '0'})
        env.start()
        self.addCleanup(env.stop)

    def respond(self, answer, message, *, history=None, text='Макет собирает показания.',
                kind='grounded', ids=None, module=None):
        calls = []
        def model(messages):
            calls.append(messages)
            return json.dumps({'answer': answer, 'answer_kind': kind,
                               'used_chunk_ids': ['C1'] if ids is None else ids,
                               'lead_offer': False}, ensure_ascii=False)
        result = (module or self.core).Consultant(FixedRetriever(text), model, competition=True).respond(
            message=message, history=history or [], brief=ClientBrief())
        return result, calls

    def test_only_consultant_changes(self):
        with ZipFile(BASE) as archive:
            self.assertEqual([name for name, value in self.payload.items()
                              if value != archive.read(name)], ['consultant.py'])

    def test_price_followup_keeps_object_and_goal(self):
        history = [ChatMessage('user', 'На складе нужен учёт арендаторов.')]
        for message in ('А теперь о цене?', 'Теперь про цену.', 'Теперь о стоимости.'):
            with self.subTest(message=message):
                query = self.core._search_query(message, history, ClientBrief())
                self.assertIn('складе', query)
                self.assertIn('арендаторов', query)
                self.assertNotIn('арендаторов', self.control._search_query(message, history, ClientBrief()))

    def test_real_topic_switch_still_discards_old_object(self):
        history = [ChatMessage('user', 'На складе нужен учёт арендаторов.')]
        for message in ('А теперь о солнечной станции.', 'Другой объект: офис.'):
            self.assertNotIn('арендаторов', self.core._search_query(message, history, ClientBrief()))

    def test_utochnu_question_is_not_a_correction(self):
        for message in ('Уточню: сколько стоит расчёт?', 'Уточню, как это работает?'):
            self.assertFalse(self.core._is_correction(message))
        self.assertTrue(self.core._is_correction('Уточню: 24 прибора. Сколько это стоит?'))
        self.assertTrue(self.core._is_correction('Уточню: это офис, а не склад. Как начать?'))

    def test_price_question_keeps_manager_summary(self):
        history = [ChatMessage('user', 'Нужен учёт по арендаторам.')]
        message = 'Уточню: сколько стоит расчёт? Хочу связаться со специалистом.'
        result, _ = self.respond('Макет собирает показания.', message, history=history)
        self.assertIn('учёт по арендаторам', result.brief.manager_summary)

    def test_quoted_role_label_does_not_destroy_ordinary_question(self):
        for quote in ('«Ассистент: готово»', '"Ассистент: готово"'):
            question = 'На экране написано ' + quote + '. Что это значит?'
            captured = []
            def model(messages):
                captured.append(json.loads(messages[1]['content']))
                return json.dumps({'answer_kind': 'clarification', 'answer': 'Что именно нужно уточнить?',
                                   'used_chunk_ids': []}, ensure_ascii=False)
            self.core.answer_test_case(question, consultant=self.core.Consultant(
                FixedRetriever(), model, competition=True))
            self.assertEqual(captured[0]['current_message'], question)
            self.assertEqual(captured[0]['history'], [])

    def test_role_labels_inside_user_quotes_are_data(self):
        captured = []
        class Core:
            def respond(self, **kwargs):
                captured.append(kwargs)
                return types.SimpleNamespace(answer='Проверено')
        self.core.answer_test_case('Пользователь: Вижу «Ассистент: ошибка».\n'
            'Ассистент: Что уточнить?\nПользователь: Объясните сообщение.', consultant=Core())
        self.assertEqual(captured[0]['message'], 'Объясните сообщение.')
        self.assertEqual(len(captured[0]['history']), 2)
        self.assertEqual(captured[0]['history'][0].content, 'Вижу «Ассистент: ошибка».')

    def test_existing_inline_and_multiline_dialogues_keep_the_last_turn(self):
        class Core:
            def respond(self, **kwargs):
                self.values = kwargs
                return types.SimpleNamespace(answer='Проверено')
        for separator in (' ', '\n'):
            core = Core()
            self.core.answer_test_case(separator.join(['Пользователь: У нас склад.',
                'Ассистент: Какая задача?', 'Пользователь: Нужен учёт.']), consultant=core)
            self.assertEqual(core.values['message'], 'Нужен учёт.')
            self.assertEqual(len(core.values['history']), 2)

    def test_contact_handoff_keeps_comparison_without_question_mark(self):
        answer = 'Макет собирает показания, а Корабль хранит архив.'
        history = [ChatMessage('user', 'Мой номер +70000000000')]
        message = 'Хочу связаться со специалистом. Сравните решения.'
        result, calls = self.respond(answer, message, history=history, text=answer)
        self.assertTrue(result.answer.startswith(answer))
        self.assertEqual(len(calls), 1)
        control, _ = self.respond(answer, message, history=history, text=answer, module=self.control)
        self.assertNotIn('Корабль', control.answer)

    def test_refusal_keeps_control_prompt_and_useful_answer(self):
        history = [ChatMessage('user', 'Телефон не оставлю.'), ChatMessage('assistant', 'Продолжим.')]
        result, calls = self.respond('Макет собирает показания.', 'Какие функции есть?',
            history=history)
        control, control_calls = self.respond('Макет собирает показания.', 'Какие функции есть?',
                                             history=history, module=self.control)
        self.assertEqual(calls, control_calls)
        self.assertEqual(result.answer, control.answer)
        self.assertEqual(result.answer, 'Макет собирает показания.')
        self.assertFalse(result.lead_offer)

    def test_generation_and_review_rules_are_byte_identical(self):
        for name in ('SYSTEM_RULES', 'COMPETITION_RULES', 'REVIEW_RULES', 'COMPLETENESS_RULES'):
            self.assertEqual(getattr(self.core, name), getattr(self.control, name))

    def test_audited_correct_answers_are_preserved_in_one_call(self):
        cases = [
            ('Макет собирает показания и хранит архив.', '1. Макет собирает показания.\n2. Макет хранит архив.', 'grounded', ['C1']),
            ('Порог макета 750 кВт.', 'Порог макета 0,75 МВт.', 'grounded', ['C1']),
            ('Макет собирает показания.', 'Количество приборов уже известно. Как сейчас собираются показания?', 'clarification', []),
            ('Производитель гарантирует сохранность архива макета.', 'Производитель гарантирует сохранность архива макета.', 'grounded', ['C1']),
        ]
        for source, answer, kind, ids in cases:
            result, calls = self.respond(answer, 'У нас 12 приборов. Расскажите о макете.',
                                        text=source, kind=kind, ids=ids)
            self.assertEqual(result.answer, answer)
            self.assertEqual(len(calls), 1)

    def test_missing_source_repair_still_runs(self):
        calls = []
        def model(messages):
            calls.append(messages)
            return json.dumps({'answer': 'Макет собирает показания.', 'answer_kind': 'grounded',
                               'used_chunk_ids': [] if len(calls) == 1 else ['C1']}, ensure_ascii=False)
        result = self.core.Consultant(FixedRetriever(), model, competition=True).respond(
            message='Что умеет макет?', history=[], brief=ClientBrief())
        self.assertEqual(result.answer, 'Макет собирает показания.')
        self.assertEqual(len(calls), 2)


if __name__ == '__main__':
    unittest.main()
