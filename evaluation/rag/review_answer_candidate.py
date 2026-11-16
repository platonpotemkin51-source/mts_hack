"""Frozen review: four invented correct answers lost in 8682b26 versus 40e8ef7."""
import sys,json,os,types,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
candidate=types.ModuleType('reviewed_consultant')
exec(subprocess.check_output(['git','show','8682b26:consultant.py'],cwd=ROOT,text=True),candidate.__dict__)
from contracts import ClientBrief,KnowledgeChunk
from unittest.mock import patch
old=types.ModuleType('control_consultant');exec(subprocess.check_output(['git','show','40e8ef7:consultant.py'],cwd=ROOT,text=True),old.__dict__)
class Retriever:
 def __init__(self,text):self.text=text
 def retrieve(self,q,top_k=6):return [KnowledgeChunk('synthetic','fiction.md','Вымышленный макет',self.text)]
cases=[
 {'id':'list_numbers','source':'Макет собирает показания и хранит архив.','question':'Какие функции есть у макета?','answer':'1. Макет собирает показания.\n2. Макет хранит архив.','kind':'grounded','ids':['C1']},
 {'id':'unit_conversion','source':'Порог мощности макета 750 кВт.','question':'Какой порог мощности у макета?','answer':'Порог мощности макета 0,75 МВт.','kind':'grounded','ids':['C1']},
 {'id':'known_count_false_positive','source':'Макет собирает показания.','question':'У нас 12 приборов. Нужен учёт.','answer':'Количество приборов уже известно. Как сейчас собираются показания?','kind':'clarification','ids':[]},
 {'id':'sourced_warranty','source':'Производитель гарантирует сохранность архива макета.','question':'Что гарантирует производитель макета?','answer':'Производитель гарантирует сохранность архива макета.','kind':'grounded','ids':['C1']},
]
rows=[]
for case in cases:
 row={'id':case['id'],'expected_answer':case['answer']}
 for label,module in [('control',old),('candidate',candidate)]:
  calls=[];events=[]
  def model(messages):
   calls.append(1)
   return json.dumps({'answer':case['answer'],'answer_kind':case['kind'],'used_chunk_ids':case['ids'],'lead_offer':False},ensure_ascii=False)
  with patch.dict(os.environ,{'ANSWER_REVIEW':'0','CLIENT_FACTS':'1','CONTEXT_MODE':'chunks'}):
   result=module.Consultant(Retriever(case['source']),model,competition=True,diagnostic_fn=lambda k,v=None:events.append(k)).respond(message=case['question'],history=[],brief=ClientBrief())
  row[label]={'answer':result.answer,'calls':len(calls),'events':events,'kept_correct_answer':result.answer==case['answer']}
 rows.append(row)
 print(json.dumps(row,ensure_ascii=False))
assert all(row['control']['kept_correct_answer'] for row in rows)
assert all(not row['candidate']['kept_correct_answer'] for row in rows)
path=ROOT/'runtime/evaluation/review-8682b26-regressions.json'
path.parent.mkdir(parents=True,exist_ok=True)
path.write_text(json.dumps(rows,ensure_ascii=False,indent=2))
print('Reproduced 4 candidate regressions; control preserves all 4 answers; no real API calls.')
