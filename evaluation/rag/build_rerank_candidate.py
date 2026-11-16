"""Build a reranker-only experiment from the verified 61.664 archive."""
import ast
import hashlib
import json
from pathlib import Path
from zipfile import ZIP_DEFLATED,ZipFile

ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'runtime/submission-hybrid-evidence.zip'
EXPECTED='23a0d691d098353f5b078b025b8ff70e4af41f196bd5be1f05200b479f8681ba'
FILES={'solution.py','requirements.txt','contracts.py','consultant.py','llm_client.py','kb_loader.py','chunking.py','retrieval.py'}

def main():
    if hashlib.sha256(BASE.read_bytes()).hexdigest()!=EXPECTED:
        raise SystemExit('Baseline archive differs from verified hybrid candidate; refusing mixed experiment')
    with ZipFile(BASE)as z:payload={name:z.read(name)for name in z.namelist()}
    if set(payload)!=FILES:raise SystemExit('Unexpected baseline archive structure')
    source=payload['solution.py'].decode('utf-8')
    anchor='os.environ.setdefault("RETRIEVAL_MODE", "hybrid")'
    if source.count(anchor)!=1:raise SystemExit('Baseline hybrid setting is missing or ambiguous')
    payload['solution.py']=source.replace(anchor,anchor+'\nos.environ.setdefault("RERANK_ENABLED", "1")').encode()
    payload['retrieval.py']=(ROOT/'retrieval.py').read_bytes()
    for name,value in payload.items():
        if name.endswith('.py'):ast.parse(value.decode('utf-8'),filename=name)
    output=ROOT/'runtime/submission-hybrid-rerank.zip'
    with ZipFile(output,'w',compression=ZIP_DEFLATED)as z:
        for name,value in payload.items():z.writestr(name,value)
    with ZipFile(output)as z:
        assert z.testzip()is None and set(z.namelist())==FILES
    unchanged=sorted(FILES-{'solution.py','retrieval.py'})
    with ZipFile(BASE)as old:
        assert all(payload[name]==old.read(name)for name in unchanged)
    meta={'baseline_sha256':EXPECTED,'candidate_sha256':hashlib.sha256(output.read_bytes()).hexdigest(),
          'unchanged_files':unchanged,'changed_files':['retrieval.py','solution.py (enable reranker only)']}
    (ROOT/'runtime/evaluation/rerank-candidate-manifest.json').write_text(json.dumps(meta,indent=2))
    print(output);print(json.dumps(meta,indent=2))

if __name__=='__main__':main()
