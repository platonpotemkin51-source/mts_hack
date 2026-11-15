"""Mentioning missing documentation is not requesting technical instructions."""
import unittest
from consultant import _out_of_scope


class ScopeIntentTests(unittest.TestCase):
    def test_missing_documentation_does_not_block_consultation(self):
        for text in ('У нас нет технической документации. Можно обсудить учёт?',
                     'Техническая документация потеряна, с чего начать оценку?',
                     'Документации нет. Нужно понять, подойдёт ли платформа.'):
            self.assertFalse(_out_of_scope(text))

    def test_actual_documentation_requests_remain_out_of_scope(self):
        for text in ('Покажите техническую документацию по прибору',
                     'Пришлите техническую документацию', 'Нужна техническая документация'):
            self.assertTrue(_out_of_scope(text))

    def test_missing_docs_do_not_enable_hardware_instructions(self):
        self.assertTrue(_out_of_scope('У нас нет документации. Как прошить счётчик?'))


if __name__ == '__main__':
    unittest.main()
