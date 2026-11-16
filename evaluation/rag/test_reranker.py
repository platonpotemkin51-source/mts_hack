"""Organizer rerank contract, dense recall, companion evidence and safe degradation."""
import json
import unittest
from unittest.mock import patch
import httpx
from contracts import KnowledgeChunk
from retrieval import BM25Retriever,HybridRetriever,OrganizerReranker,RerankerError

class Fixed:
    def __init__(self,values):self.values,self.calls=values,0
    def retrieve(self,query,top_k=6):self.calls+=1;return self.values[:top_k]

class RerankerTests(unittest.TestCase):
    def setUp(self):
        self.env=patch.dict('os.environ',{'LLM_BASE_URL':'http://organizer.test/v1','LLM_API_KEY':'secret-placeholder',
                                        'RERANKER_API_KEY':'rerank-placeholder','RERANKER_MODEL':'test-reranker'},clear=True)
        self.env.start();self.addCleanup(self.env.stop)
        self.chunks=[KnowledgeChunk(str(i),'20-products/paper.md','Макет / Раздел','Макет: текст '+str(i))for i in range(25)]
        self.requests=[];self.response=httpx.Response(200,json={'results':[{'index':2},{'index':0}]})
        original=httpx.Client
        def handler(request):
            self.requests.append(request)
            if isinstance(self.response,Exception):raise self.response
            return self.response
        self.p=patch('httpx.Client',side_effect=lambda **kw:original(transport=httpx.MockTransport(handler),**kw))
        self.p.start();self.addCleanup(self.p.stop)

    def test_request_contract_and_index_mapping(self):
        r=OrganizerReranker();self.assertEqual(self.requests,[])
        result=r.rerank('Вопрос',self.chunks[:3],2)
        self.assertEqual([c.chunk_id for c in result],['2','0'])
        self.assertEqual([c.score for c in result],[1.0,0.5])
        req=self.requests[0];body=json.loads(req.content)
        self.assertEqual(str(req.url),'http://organizer.test/v1/rerank')
        self.assertEqual(req.headers['Authorization'],'Bearer rerank-placeholder')
        self.assertEqual(body,{'model':'test-reranker','query':'Вопрос','documents':[c.title+'\n'+c.text for c in self.chunks[:3]],'top_n':2})
        self.assertEqual(req.extensions['timeout']['read'],30.0)

    def test_data_response_and_single_candidate(self):
        self.response=httpx.Response(200,json={'data':[{'index':0}]})
        result=OrganizerReranker().rerank('Вопрос',self.chunks[:1],6)
        self.assertEqual(result[0].chunk_id,self.chunks[0].chunk_id)

    def test_invalid_indices_and_short_or_malformed_results_are_errors(self):
        for value in (None,{}, {'results':[]},{'results':[{'index':True}]},
                      {'results':[{'index':-1}]},{'results':[{'index':99}]},
                      {'results':[{'index':'1.5'}]},{'results':[{}]},
                      {'results':[{'index':0},{'index':0}]}):
            with self.subTest(value=value):
                self.response=httpx.Response(200,json=value)
                with self.assertRaises(RerankerError):OrganizerReranker().rerank('Вопрос',self.chunks[:3],1)

    def test_numeric_string_indices_match_the_organizer_example(self):
        self.response=httpx.Response(200,json={'results':[{'index':' 2 '},{'index':'0'}]})
        result=OrganizerReranker().rerank('Вопрос',self.chunks[:3],2)
        self.assertEqual([c.chunk_id for c in result],['2','0'])

    def test_failures_are_sanitized_and_do_not_retry_storm(self):
        for value in (httpx.Response(401,text='secret-placeholder PRIVATE'),httpx.Response(429),
                      httpx.Response(503),httpx.ReadTimeout('secret-placeholder PRIVATE')):
            self.response=value;self.requests.clear()
            with self.assertRaises(RerankerError)as caught:OrganizerReranker().rerank('Вопрос',self.chunks,2)
            self.assertNotIn('secret-placeholder',str(caught.exception));self.assertNotIn('PRIVATE',str(caught.exception))
            self.assertEqual(len(self.requests),1)

    def test_hybrid_reranks_twenty_candidates_once(self):
        lexical,semantic=Fixed(self.chunks),Fixed(self.chunks)
        h=HybridRetriever(lexical,semantic,reranker=OrganizerReranker())
        result=h.retrieve('Вопрос',2)
        body=json.loads(self.requests[0].content)
        self.assertEqual(len(body['documents']),20);self.assertEqual(len(result),2)
        self.assertEqual(len(self.requests),1);self.assertEqual(semantic.calls,1)
        self.assertEqual(result[0].text,body['documents'][2].split('\n',1)[1])

    def test_unavailable_reranker_keeps_the_complete_hybrid_order(self):
        expected=HybridRetriever(Fixed(self.chunks),Fixed(self.chunks)).retrieve('Вопрос',6)
        self.response=httpx.Response(500,text='PRIVATE')
        with self.assertLogs('retrieval',level='WARNING')as logs:
            result=HybridRetriever(Fixed(self.chunks),Fixed(self.chunks),reranker=OrganizerReranker()).retrieve('Вопрос',6)
        self.assertEqual(result,expected);self.assertNotIn('PRIVATE',' '.join(logs.output))

    def test_reranked_answer_keeps_its_companion_limitation(self):
        chunks=[KnowledgeChunk('intro','20-products/paper.md','Макет / Что это','Макет сканирует бумагу.'),
                KnowledgeChunk('answer','20-products/paper.md','Макет / Возможности','Макет читает листы.'),
                KnowledgeChunk('limit','20-products/paper.md','Макет / Что мы не делаем','Макет не гарантирует точность.')]
        self.response=httpx.Response(200,json={'results':[{'index':1}]})
        semantic=Fixed(chunks)
        h=HybridRetriever(BM25Retriever(chunks),semantic,reranker=OrganizerReranker())
        found=h.retrieve_with_evidence('необычныйзапрос',1)
        self.assertIn('limit',{c.chunk_id for c in found});self.assertEqual(semantic.calls,1)
        self.assertEqual(len(self.requests),1)

    def test_empty_query_and_empty_candidates_do_not_call_service(self):
        self.assertEqual(OrganizerReranker().rerank('Вопрос',[],6),[])
        self.assertEqual(OrganizerReranker().rerank('',self.chunks,6),[])
        self.assertEqual(HybridRetriever(Fixed([]),Fixed([]),reranker=OrganizerReranker()).retrieve('',6),[])
        self.assertEqual(self.requests,[])

if __name__=='__main__':unittest.main()
