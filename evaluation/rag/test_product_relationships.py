"""Explicit complementary-product evidence is retained only for mixed input tasks."""
import unittest
from unittest.mock import patch
from contracts import KnowledgeChunk
from retrieval import BM25Retriever

class RelationshipTests(unittest.TestCase):
    def retriever(self, relation='Фото дополняет [учёт](meter.md), но не заменяет его.'):
        self.chunks=[KnowledgeChunk('meter','20-products/meter.md','Учёт / Что это','Учёт собирает показания приборов.'),
                     KnowledgeChunk('photo','20-products/photo.md','Фото / Что это','Фото читает показания.'),
                     KnowledgeChunk('meter-extra','20-products/meter.md','Учёт / Возможности','Учёт приборов.'),
                     KnowledgeChunk('photo-extra','20-products/photo.md','Фото / Возможности','Фото приборов.'),
                     KnowledgeChunk('relation','20-products/photo.md','Фото / Назначение',relation)]
        return BM25Retriever(self.chunks)

    def context(self,r,query):
        with patch.object(r,'_character_candidates',return_value=[]):
            return r.retrieve_with_evidence(query,4,ranked=list(r._chunks[:4]))

    def test_explicit_relationship_is_added_without_replacing_existing_evidence(self):
        r=self.retriever();before=list(r._chunks[:4]);found=self.context(r,'Часть приборов подключена, часть показаний получаем по фото')
        self.assertEqual(found[:4],before)
        self.assertEqual(found[-1].chunk_id,'relation')
        self.assertEqual(found[-1].text,self.chunks[-1].text)
        self.assertEqual(len(found),len({c.chunk_id for c in found}))

    def test_ordinary_price_or_privacy_question_does_not_add_relationship(self):
        r=self.retriever()
        for q in ('Сколько стоит учёт?','Обрабатываются ли персональные данные?','Какова точность фото?'):
            self.assertNotIn('relation',{c.chunk_id for c in self.context(r,q)})

    def test_missing_external_or_unrelated_link_does_not_create_a_relationship(self):
        for target in ('missing.md','https://example.invalid/meter.md','../../secret.md'):
            r=self.retriever('Фото дополняет [учёт]('+target+').')
            self.assertNotIn('relation',{c.chunk_id for c in self.context(r,'Фото и подключённые приборы')})

    def test_long_relation_is_not_truncated_or_added(self):
        r=self.retriever('Фото дополняет [учёт](meter.md). '+'Условие '*200)
        self.assertNotIn('relation',{c.chunk_id for c in self.context(r,'Фото и подключённые приборы')})

    def test_a_navigation_link_alone_is_not_a_statement_of_complementarity(self):
        r=self.retriever('Смотрите [учёт](meter.md).')
        self.assertNotIn('relation',{c.chunk_id for c in self.context(r,'Фото и подключённые приборы')})

if __name__=='__main__':unittest.main()
