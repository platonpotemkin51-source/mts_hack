"""The submission package is flat, minimal, and independent of local secrets."""
import contextlib
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from zipfile import ZipFile

from scripts import build_submission


class PackagingTests(unittest.TestCase):
    def test_flat_package_excludes_secrets_and_preserves_core(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for name in build_submission.FILES:
                (root / name).write_bytes((build_submission.ROOT / name).read_bytes())
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
                        self.assertEqual(archive.read(name), (root / name).read_bytes())
