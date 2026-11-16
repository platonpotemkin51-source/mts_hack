"""Whole documents retain distant caveats without replacing unrelated evidence."""
import json
import unittest
from unittest.mock import patch
from chunking import chunk_documents
from contracts import KnowledgeDocument, ClientBrief
from retrieval import BM25Retriever
from consultant import Consultant

class DocumentContextTests(unittest.TestCase):
    def corpus(self, tail='Оговорка: результат не гарантируется.'):
        return [KnowledgeDocument('a','20-products/a.md','Макет',
            '# Макет\n## Что это\nСканирование бумаги.\n## Возможности\nСканирование листов бумаги.\n## Условия\n'+tail),
            KnowledgeDocument('b','00-reference/ref.md','Справка','# Справка\nПроверка бумаги обязательна.')]

    def test_intact_document_and_unrelated_evidence_are_retained(self):
        docs=self.corpus();r=BM25Retriever(chunk_documents(docs),documents=docs)
        old=r.retrieve_with_evidence('сканирование бумаги проверка',3)
        new=r.retrieve_with_documents('сканирование бумаги проверка',3)
        self.assertTrue(any(c.text==docs[0].text for c in new))
        self.assertTrue(any('результат не гарантируется' in c.text for c in new))
        self.assertTrue(all(any(n.chunk_id==c.chunk_id for n in new) for c in old if c.source!=docs[0].source))
        self.assertEqual(len(new),len({c.chunk_id for c in new}))
        self.assertEqual(new,r.retrieve_with_documents('сканирование бумаги проверка',3))

    def test_oversized_document_keeps_original_fragments_without_truncation(self):
        docs=self.corpus('OTHER ' * 9000);r=BM25Retriever(chunk_documents(docs),documents=docs)
        before=r.retrieve_with_evidence('сканирование бумаги',2)
        self.assertEqual(before,r.retrieve_with_documents('сканирование бумаги',2))

    def test_empty_and_invalid_queries_preserve_contract(self):
        docs=self.corpus();r=BM25Retriever(chunk_documents(docs),documents=docs)
        self.assertEqual(r.retrieve_with_documents(''),[])
        with self.assertRaises(ValueError):r.retrieve_with_documents('бумага',0)

    def test_whole_document_gets_a_valid_citation_in_the_shared_core(self):
        docs=self.corpus();r=BM25Retriever(chunk_documents(docs),documents=docs);captured=[]
        def model(messages):
            supplied=json.loads(messages[1]['content']);captured.extend(supplied['knowledge_chunks'])
            return json.dumps({'answer_kind':'grounded','answer':'Результат не гарантируется.',
                               'used_chunk_ids':['C1'],'lead_offer':False})
        with patch.dict('os.environ',{'CONTEXT_MODE':'documents'}):
            result=Consultant(r,model,competition=True,context_expansion=True).respond(
                message='Сканирование бумаги',history=[],brief=ClientBrief())
        self.assertEqual(captured[0]['text'],docs[0].text)
        self.assertEqual(result.sources[0].text,docs[0].text)

if __name__=='__main__':unittest.main()
