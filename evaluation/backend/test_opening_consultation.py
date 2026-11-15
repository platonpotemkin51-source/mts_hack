"""Natural dialogue survives without product citations; factual claims stay guarded."""
import json
import unittest
from consultant import Consultant, InvalidModelAnswer, _parse_model_answer
from contracts import ChatMessage, ClientBrief, KnowledgeChunk
from llm_client import ModelUnavailable


class Retriever:
    def retrieve(self, query, top_k=6):
        return [KnowledgeChunk('doc', 'invented.md', 'Макет', 'Макет собирает показания.')]


def payload(answer, kind='clarification', ids=None):
    return json.dumps({'answer_kind':kind,'answer':answer,'used_chunk_ids':ids or [],
                       'lead_offer':False,'brief':{'object':None,'problem':None,'known_parameters':[],
                                                 'recommended_products':[],'missing_fields':[]}})


class OpeningTests(unittest.TestCase):
    def respond(self, replies, message='Нужна консультация', history=None):
        calls=[]
        def model(messages):
            calls.append(messages)
            value=replies[len(calls)-1]
            if isinstance(value, Exception): raise value
            return value
        result=Consultant(Retriever(),model).respond(message=message,history=history or [],brief=ClientBrief())
        return result,calls

    def test_natural_clarifications_are_preserved_verbatim(self):
        for answer in ('Здравствуйте! Расскажите, какой объект хотите обсудить.',
                       'Понял, склад. Что изменилось: потребление или сумма платежа?',
                       'Спасибо, картину понял. Можете прислать данные из последнего счёта?',
                       'Продолжим без телефона. Опишите, как сейчас собираются показания.'):
            with self.subTest(answer=answer):
                result,calls=self.respond([payload(answer)],message='Склад',
                    history=[ChatMessage('user','Помоги разобраться с расходами'),
                             ChatMessage('assistant','Какой у вас объект?')])
                self.assertEqual(result.answer,answer)
                self.assertEqual(result.sources,[])
                self.assertEqual(len(calls),1)

    def test_client_numbers_do_not_require_knowledge_sources(self):
        answer='На складе 37 приборов, понял. Как сейчас получаете их показания?'
        result,calls=self.respond([payload(answer)],message='У нас склад с 37 приборами.')
        self.assertEqual(result.answer,answer)
        self.assertEqual(len(calls),1)

    def test_specific_limitation_is_preserved_without_a_canned_refusal(self):
        answer='Поддержка этого протокола в материалах не подтверждена. Какие приборы вы используете?'
        result,calls=self.respond([payload(answer,'limitation')],message='Поддерживается ли тестовый протокол?')
        self.assertEqual(result.answer,answer)
        self.assertEqual(len(calls),1)

    def test_unclassified_natural_dialogue_is_repaired_once_instead_of_replaced(self):
        answer='Понял, склад. Чтобы разобраться в ситуации, расскажите, что изменилось в счетах.'
        draft=json.loads(payload(answer));draft.pop('answer_kind')
        result,calls=self.respond([json.dumps(draft),payload(answer)],message='Склад')
        self.assertEqual(result.answer,answer)
        self.assertEqual(len(calls),2)

    def test_missing_citation_is_recovered_without_enabling_optional_review(self):
        result,calls=self.respond([payload('Макет собирает показания.','grounded'),
                                  payload('Макет собирает показания.','grounded',['C1'])])
        self.assertEqual(result.answer,'Макет собирает показания.')
        self.assertEqual(result.sources[0].chunk_id,'doc')
        self.assertEqual(len(calls),2)

    def test_dialogue_label_does_not_authorize_new_numeric_guarantees(self):
        safe='Гарантированный процент экономии не указан. Какую задачу хотите решить?'
        result,calls=self.respond([payload('Гарантируем экономию 91%.'),payload(safe,'limitation')],
                                 message='Какая экономия?')
        self.assertNotIn('91',result.answer)
        self.assertEqual(result.answer,safe)
        self.assertEqual(len(calls),2)

    def test_failed_repair_does_not_pass_through_unverified_product_facts(self):
        result,calls=self.respond([payload('Макет бесплатен.','grounded',['C99']),ModelUnavailable('timeout')],
                                 message='Сколько стоит макет?')
        self.assertNotIn('бесплатен',result.answer)
        self.assertEqual(result.sources,[])
        self.assertEqual(len(calls),2)

    def test_unknown_answer_kind_is_rejected(self):
        with self.assertRaises(InvalidModelAnswer):
            _parse_model_answer(payload('Ответ.','arbitrary'),competition=True)

    def test_internal_filenames_are_not_passed_as_answer_material(self):
        result,calls=self.respond([payload('Что именно хотите выяснить?')])
        supplied=json.loads(calls[0][1]['content'])
        self.assertNotIn('invented.md',json.dumps(supplied['knowledge_chunks']))
        self.assertEqual(supplied['knowledge_chunks'][0]['chunk_id'],'C1')


if __name__=='__main__':
    unittest.main()
