"""The submission package is flat, minimal, and independent of local secrets."""
import contextlib
import io
import tempfile
import subprocess
import sys
import unittest
from pathlib import Path
from unittest.mock import patch
from zipfile import ZipFile

from scripts import build_submission


class PackagingTests(unittest.TestCase):
    def test_flat_package_excludes_secrets_and_preserves_core(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for path in build_submission.FILES.values():
                (root / path).parent.mkdir(parents=True, exist_ok=True)
                (root / path).write_bytes((build_submission.ROOT / path).read_bytes())
            (root / '.env').write_text('LLM_API_KEY=synthetic-secret')
            with patch.object(build_submission, 'ROOT', root), contextlib.redirect_stdout(io.StringIO()):
                build_submission.main()
            with ZipFile(root / 'runtime/submission.zip') as archive:
                self.assertEqual(set(archive.namelist()), set(build_submission.FILES))
                self.assertEqual(archive.read('requirements.txt'), b'httpx==0.28.1\n')
                for name in archive.namelist():
                    self.assertNotIn('/', name)
                    self.assertNotIn(b'synthetic-secret', archive.read(name))
                    if name.endswith('.py'):
                        compile(archive.read(name), name, 'exec')
                        self.assertEqual(archive.read(name), build_submission.submission_file(name))
                        self.assertNotIn(b"from energytool.", archive.read(name))
                    (root / name).write_bytes(archive.read(name))
            # Run outside the checkout so package imports cannot mask a broken ZIP.
            result = subprocess.run([sys.executable, "-I", "-c",
                "import sys; sys.path.insert(0, '.'); import solution; import consultant; "
                "import retrieval; assert callable(solution.run)"],
                cwd=root, capture_output=True, text=True, timeout=20)
            self.assertEqual(result.returncode, 0, result.stderr)
