"""Dense-only candidates retain lexical companion limitations with one query embedding."""
import json
import unittest
from unittest.mock import patch
from energytool.consultant import Consultant
from energytool.contracts import ClientBrief, KnowledgeChunk
from energytool.retrieval import BM25Retriever, HybridRetriever

class Semantic:
    def __init__(self, result): self.result,self.calls=result,0
    def retrieve(self, query, top_k=6): self.calls+=1;return self.result[:top_k]

class HybridEvidenceTests(unittest.TestCase):
    def corpus(self):
        source='20-products/paper.md'
        return [KnowledgeChunk('intro',source,'Бумажный макет / Что это','Бумажный макет хранит бумагу.'),
                KnowledgeChunk('answer',source,'Бумажный макет / Возможности','Бумажный макет сканирует листы.'),
                KnowledgeChunk('limit',source,'Бумажный макет / Что мы не делаем','Бумажный макет не гарантирует точность.')]

    def test_dense_only_hit_gets_its_missing_limitation(self):
        chunks=self.corpus();lexical=BM25Retriever(chunks);semantic=Semantic([chunks[1]])
        self.assertEqual(lexical.retrieve('необычнаяформулировка'),[])
        hybrid=HybridRetriever(lexical,semantic)
        found=hybrid.retrieve_with_evidence('необычнаяформулировка',1)
        self.assertEqual(found[0].chunk_id,'answer')
        self.assertIn('limit',{c.chunk_id for c in found})
        self.assertEqual(len(found),len({c.chunk_id for c in found}))
        self.assertEqual(semantic.calls,1)
        self.assertEqual(len(hybrid.retrieve('необычнаяформулировка',1)),1)

    def test_companion_limitations_reach_the_model(self):
        chunks=self.corpus();semantic=Semantic([chunks[1]]);captured=[]
        def model(messages):
            data=json.loads(messages[1]['content']);captured.extend(data['knowledge_chunks'])
            return json.dumps({'answer_kind':'grounded','answer':'Точность не гарантируется.','used_chunk_ids':['C1']})
        with patch.dict('os.environ',{'CONTEXT_MODE':'chunks','EVIDENCE_COMPLETION':'1'}):
            Consultant(HybridRetriever(BM25Retriever(chunks),semantic),model,competition=True,
                       context_expansion=True).respond(message='необычнаяформулировка',history=[],brief=ClientBrief())
        self.assertTrue(any('не гарантирует точность' in c['text']for c in captured))
        self.assertEqual(semantic.calls,1)

    def test_empty_query_does_not_call_embeddings(self):
        semantic=Semantic(self.corpus());h=HybridRetriever(BM25Retriever(self.corpus()),semantic)
        self.assertEqual(h.retrieve_with_evidence(''),[])
        self.assertEqual(semantic.calls,0)

if __name__=='__main__':unittest.main()
