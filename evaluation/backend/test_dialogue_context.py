"""Context regressions on invented client statements, without a live model."""
import json
import unittest

from energytool.consultant import Consultant, _active_users, _explicit_lead_request, _model_history, _safe_unsourced_answer, _search_query
from energytool.contracts import ChatMessage, ClientBrief


class EmptyRetriever:
    def retrieve(self, query, top_k=6):
        return []


class DialogueTests(unittest.TestCase):
    def test_manager_and_imperative_contact_requests(self):
        for message in ("Хочу связаться с менеджером", "Перезвоните мне", "Свяжите меня со специалистом", "Передайте обращение менеджеру"):
            with self.subTest(message=message):
                self.assertTrue(_explicit_lead_request(message))

    def test_negated_contact_imperatives_do_not_create_lead(self):
        for message in ("Не хочу связаться с менеджером", "Не перезвоните мне", "Не позвоните мне", "Не передайте обращение менеджеру", "Не свяжите меня со специалистом"):
            with self.subTest(message=message):
                self.assertFalse(_explicit_lead_request(message))

    def test_known_count_is_not_asked_again(self):
        result = _safe_unsourced_answer("Сколько приборов у вас?", client_text="У нас 17 приборов.")
        self.assertEqual(result, "Какой у вас тип объекта?")
        result = _safe_unsourced_answer("Сколько приборов у вас?", client_text="На складе 17 приборов.")
        self.assertEqual(result, "Как сейчас собираются показания?")

    def test_models_and_collection_are_not_rewritten_as_quantity(self):
        self.assertEqual(_safe_unsourced_answer("Какие модели приборов у вас?"),
                         "Какие модели приборов у вас установлены?")
        self.assertEqual(_safe_unsourced_answer("Как собираются показания счётчиков?"),
                         "Как сейчас собираются показания?")

    def test_known_collection_and_object_are_not_repeated(self):
        facts = "У нас склад и 17 приборов. Собираем показания вручную."
        self.assertEqual(_safe_unsourced_answer("Какой тип объекта?", client_text=facts),
                         "В каком регионе находится объект?")

    def test_unknown_count_is_still_asked(self):
        self.assertEqual(_safe_unsourced_answer("Сколько приборов у вас?", client_text="Количество пока не знаем."),
                         "Сколько точек учёта у вас?")

    def test_assistant_numbers_are_not_client_facts(self):
        raw = json.dumps({"answer": "Сколько приборов у вас?", "used_chunk_ids": [], "lead_offer": False})
        result = Consultant(EmptyRetriever(), lambda _: raw, competition=True).respond(
            message="Нужно обсудить учёт", history=[ChatMessage("assistant", "Возможно, у вас 17 приборов?")], brief=ClientBrief())
        self.assertEqual(result.answer, "Сколько точек учёта у вас?")

    def test_real_core_uses_user_history_for_safe_question(self):
        raw = json.dumps({"answer": "Сколько приборов у вас?", "used_chunk_ids": [], "lead_offer": False})
        result = Consultant(EmptyRetriever(), lambda _: raw, competition=True).respond(
            message="Какие ещё сведения нужны?", history=[ChatMessage("user", "На складе 17 приборов.")], brief=ClientBrief())
        self.assertEqual(result.answer, "Как сейчас собираются показания?")

    def test_correction_removes_old_count_from_history_and_brief_query(self):
        query = _search_query("Исправляю: 24 прибора.", [ChatMessage("user", "На складе 17 приборов, нужен учёт.")],
                              ClientBrief(known_parameters=["17 приборов"]))
        self.assertNotIn("17", query)
        self.assertIn("24", query)
        self.assertIn("складе", query)
        self.assertIn("учёт", query)

    def test_inverted_correction_and_power_unit_conversion(self):
        query = _search_query("Уточню: не 17 приборов, а 24.", [ChatMessage("user", "17 приборов")], ClientBrief())
        self.assertEqual(query.count("17"), 1)  # only the explicit negation in the current message
        query = _search_query("Исправляю: 1 МВт.", [ChatMessage("user", "Мощность 1 кВт.")], ClientBrief())
        self.assertNotIn("1 кВт", query)
        self.assertIn("1 МВт", query)
        query = _search_query("Поправка: 2 000 кВт.", [ChatMessage("user", "Мощность 1 000 кВт.")], ClientBrief())
        self.assertNotIn("1 000", query)

    def test_correction_does_not_remove_unrelated_parameters(self):
        query = _search_query("Исправляю: 24 прибора.", [ChatMessage("user", "2 склада, 17 приборов, мощность 60 кВт.")], ClientBrief())
        self.assertIn("2 склада", query)
        self.assertIn("60 кВт", query)
        self.assertNotIn("17 приборов", query)

    def test_old_value_after_instead_does_not_restore_obsolete_query(self):
        query = _search_query('Исправляю: 24 прибора вместо 17 приборов.',
                              [ChatMessage('user', 'На складе 17 приборов.')], ClientBrief())
        self.assertEqual(query.count('17 приборов'), 1)  # only the explicit old value in the current statement

    def test_old_topic_is_not_reintroduced_on_later_turn(self):
        history = [ChatMessage("user", "На заводе нужен оптовый рынок."),
                   ChatMessage("user", "А теперь другой вопрос: солнечная станция."),
                   ChatMessage("assistant", "Какая цель?"), ChatMessage("user", "Резерв питания.")]
        query = _search_query("Сколько места потребуется?", history, ClientBrief())
        self.assertNotIn("оптовый", query)
        self.assertIn("солнечная", query)

    def test_now_count_change_does_not_erase_goal_or_object(self):
        history = [ChatMessage("user", "На складе 17 приборов, нужен учёт по арендаторам.")]
        query = _search_query("А теперь стало 24 прибора.", history, ClientBrief())
        self.assertIn("арендаторам", query)
        self.assertIn("складе", query)
        self.assertNotIn("17", query)

    def test_now_follow_up_does_not_erase_unchanged_goal_from_summary(self):
        raw = json.dumps({"answer": "Какую задачу вы хотите решить?", "used_chunk_ids": [],
                          "lead_offer": False, "manager_evidence": ["Нужен учёт по арендаторам."]})
        result = Consultant(EmptyRetriever(), lambda _: raw, competition=True).respond(
            message="А теперь какое решение вы предложите?",
            history=[ChatMessage("user", "Нужен учёт по арендаторам.")], brief=ClientBrief())
        self.assertIn("арендаторам", result.brief.manager_summary)

    def test_now_tell_me_about_other_product_still_switches_topic(self):
        query = _search_query("А теперь расскажите про солнечные панели.",
                              [ChatMessage("user", "Оптовый рынок для завода")], ClientBrief())
        self.assertNotIn("Оптовый", query)

    def test_early_goal_is_kept_beyond_last_three_turns(self):
        history = [ChatMessage("user", "Нужен учёт по арендаторам бизнес-центра.")]
        history += [ChatMessage("user", f"Дополнение {i}") for i in range(5)]
        query = _search_query("С чего начать?", history, ClientBrief())
        self.assertIn("арендаторам", query)

    def test_model_gets_early_facts_beyond_twelve_messages(self):
        history = [ChatMessage("user", "У объекта редкая крыша.")]
        history += [ChatMessage("assistant", "Уточнение") for _ in range(15)]
        self.assertEqual(_model_history(history)[0]["content"], history[0].content)
        self.assertEqual(len(_model_history(history)), 16)

    def test_history_budget_keeps_whole_recent_messages(self):
        history = [ChatMessage("user", str(i) + "x" * 7998) for i in range(10)]
        result = _model_history(history)
        self.assertLessEqual(sum(len(x["content"]) for x in result), 24000)
        self.assertEqual(result[-1]["content"], history[-1].content)
        self.assertTrue(all(x["content"] in [m.content for m in history] for x in result))

    def test_new_topic_can_ask_for_new_object_count(self):
        facts = "\n".join(_active_users([ChatMessage("user", "17 приборов на складе")], "Новый объект: офис"))
        self.assertEqual(_safe_unsourced_answer("Сколько приборов у вас?", client_text=facts),
                         "Сколько точек учёта у вас?")

    def test_count_correction_keeps_unchanged_goal_in_manager_summary(self):
        raw = json.dumps({"answer": "Какую задачу вы хотите решить?", "used_chunk_ids": [],
                          "lead_offer": False, "manager_evidence": ["Нужен учёт по арендаторам.", "17 приборов"]})
        result = Consultant(EmptyRetriever(), lambda _: raw, competition=True).respond(
            message="Исправляю: 24 прибора.", history=[ChatMessage("user", "Нужен учёт по арендаторам."),
                                                     ChatMessage("user", "На складе 17 приборов.")], brief=ClientBrief())
        self.assertIn("учёт по арендаторам", result.brief.manager_summary)
        self.assertIn("24", result.brief.manager_summary)
        self.assertNotIn("17", result.brief.manager_summary)

    def test_topic_switch_does_not_keep_previous_goal_in_summary(self):
        raw = json.dumps({"answer": "Какую задачу вы хотите решить?", "used_chunk_ids": [],
                          "lead_offer": False, "manager_evidence": ["Нужен учёт по арендаторам."]})
        result = Consultant(EmptyRetriever(), lambda _: raw, competition=True).respond(
            message="Другой объект: офис. Исправляю: 24 прибора.",
            history=[ChatMessage("user", "Нужен учёт по арендаторам.")], brief=ClientBrief())
        self.assertNotIn("арендаторам", result.brief.manager_summary or "")


if __name__ == "__main__":
    unittest.main()
