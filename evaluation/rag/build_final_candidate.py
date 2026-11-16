"""Apply only the reviewed dialogue patch to the preserved 61.664 ZIP."""
import ast
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / 'runtime/submission-hybrid-evidence.zip'
EXPECTED = '23a0d691d098353f5b078b025b8ff70e4af41f196bd5be1f05200b479f8681ba'
PATCH = Path(__file__).with_name('final_dialogue.patch')
FILES = {'solution.py', 'requirements.txt', 'contracts.py', 'consultant.py',
         'llm_client.py', 'kb_loader.py', 'chunking.py', 'retrieval.py'}


def candidate_payload():
    if hashlib.sha256(BASE.read_bytes()).hexdigest() != EXPECTED:
        raise ValueError('Control ZIP does not match the scored 61.664 archive')
    with ZipFile(BASE) as archive:
        if set(archive.namelist()) != FILES or len(archive.namelist()) != len(FILES):
            raise ValueError('Unexpected control ZIP structure')
        payload = {name: archive.read(name) for name in FILES}
    with tempfile.TemporaryDirectory() as folder:
        target = Path(folder) / 'consultant.py'
        target.write_bytes(payload['consultant.py'])
        subprocess.run(['git', 'apply', '--check', str(PATCH)], cwd=folder, check=True)
        subprocess.run(['git', 'apply', str(PATCH)], cwd=folder, check=True)
        payload['consultant.py'] = target.read_bytes()
    for name, content in payload.items():
        if name.endswith('.py'):
            ast.parse(content.decode('utf-8'), filename=name)
    return payload


def main():
    payload = candidate_payload()
    output = ROOT / 'runtime/submission-hybrid-dialogue-fixes.zip'
    with ZipFile(BASE) as control, ZipFile(output, 'w') as candidate:
        for entry in control.infolist():
            candidate.writestr(entry, payload[entry.filename])
    with ZipFile(BASE) as control, ZipFile(output) as candidate:
        assert candidate.testzip() is None
        changed = sorted(name for name in FILES if candidate.read(name) != control.read(name))
        assert changed == ['consultant.py']
    manifest = {'baseline_sha256': EXPECTED,
                'candidate_sha256': hashlib.sha256(output.read_bytes()).hexdigest(),
                'changed_files': changed, 'unchanged_files': sorted(FILES - set(changed))}
    report = ROOT / 'runtime/evaluation/final-candidate-manifest.json'
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(manifest, indent=2))
    print(output)
    print(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()
