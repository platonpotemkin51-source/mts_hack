"""Client field hints must come from actual user spans."""
import json
import unittest
from unittest.mock import patch
from consultant import Consultant, _client_fields
from contracts import ChatMessage, ClientBrief, KnowledgeChunk


class ClientFieldTests(unittest.TestCase):
    def test_known_object_and_corrected_count_are_explicit_quotes(self):
        history = [ChatMessage('user', 'Для офиса нужен учёт 17 приборов.')]
        fields = _client_fields(history, 'Исправляю: 24 прибора, а не 17.')
        self.assertIn({'field': 'тип объекта', 'user_quote': 'офиса'}, fields)
        self.assertIn({'field': 'количество приборов', 'user_quote': '24 прибора'}, fields)

    def test_negative_correction_remains_a_whole_quote(self):
        fields = _client_fields([], 'У нас не 17 приборов, а 24.')
        self.assertIn({'field': 'количество приборов', 'user_quote': 'не 17 приборов, а 24'}, fields)

    def test_old_number_with_repeated_unit_is_not_promoted_to_current_fact(self):
        for message in ('Исправляю: 24 прибора вместо 17 приборов.',
                        'Теперь 24 прибора, а не 17 приборов.'):
            fields = _client_fields([ChatMessage('user', 'У нас 17 приборов.')], message)
            self.assertIn({'field': 'количество приборов', 'user_quote': '24 прибора'}, fields)

    def test_assistant_hypotheticals_and_old_topics_are_not_facts(self):
        history = [ChatMessage('assistant', 'У нас склад, 50 приборов.'),
                   ChatMessage('user', 'Если для офиса понадобится 17 приборов?')]
        self.assertEqual(_client_fields(history, 'Есть вопрос'), [])
        history = [ChatMessage('user', 'У нас склад, 17 приборов.')]
        self.assertEqual(_client_fields(history, 'Другая тема: цена фото'), [])

    def test_collection_and_models_are_not_invented(self):
        fields = _client_fields([], 'Установлены Zeta-4. Собираем показания вручную.')
        self.assertIn({'field': 'модели приборов', 'user_quote': 'Zeta-4'}, fields)
        self.assertIn({'field': 'способ сбора показаний', 'user_quote': 'Собираем показания вручную'}, fields)

    def test_optional_hints_do_not_change_the_api_contract(self):
        class Retriever:
            def retrieve(self, query, top_k=6):
                return []
        captured = []
        def model(messages):
            captured.append(json.loads(messages[1]['content']))
            return json.dumps({'answer': 'Как сейчас собираются показания?', 'used_chunk_ids': [], 'lead_offer': False})
        with patch.dict('os.environ', {'CLIENT_FACTS': '1'}):
            Consultant(Retriever(), model, competition=True).respond(
                message='На складе 17 приборов.', history=[], brief=ClientBrief())
        self.assertEqual(len(captured[0]['already_known_client_fields']), 2)

    def test_direct_contact_does_not_get_blocked_by_technical_questionnaire(self):
        class Retriever:
            def retrieve(self, query, top_k=6):
                return [KnowledgeChunk('doc', 'invented.md', 'Учёт', 'Нужны модели приборов.')]
        def model(messages):
            return json.dumps({'answer': 'Уточните сначала модели ваших приборов.', 'used_chunk_ids': ['C1'],
                               'lead_offer': True, 'manager_evidence': ['Нужен учёт по арендаторам.']})
        result = Consultant(Retriever(), model, competition=True).respond(
            message='Перезвоните: +70000000000, Тестовый Клиент, Макет. Согласен на связь.',
            history=[ChatMessage('user', 'Нужен учёт по арендаторам.')], brief=ClientBrief())
        self.assertIn('номер указан', result.answer)
        self.assertIn('арендаторам', result.answer)
        self.assertNotIn('сначала модели', result.answer)
        self.assertNotIn('отправлена', result.answer)

    def test_contact_with_product_question_preserves_substantive_answer(self):
        class Retriever:
            def retrieve(self, query, top_k=6):
                return [KnowledgeChunk('doc', 'invented.md', 'Учёт', 'Тестовая цена 250 руб.')]
        for message in ('Перезвоните +70000000000. Сколько стоит?',
                        'Перезвоните +70000000000 и расскажите стоимость'):
            result = Consultant(Retriever(), lambda _: json.dumps({
                'answer': 'Тестовая цена 250 руб.', 'used_chunk_ids': ['C1'], 'lead_offer': False}), competition=True).respond(
                    message=message, history=[], brief=ClientBrief())
            self.assertIn('250', result.answer)


if __name__ == '__main__':
    unittest.main()
