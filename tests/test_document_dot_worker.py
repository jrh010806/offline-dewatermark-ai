import contextlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from tools.run_document_dot_experiment import main


class DocumentDotWorkerArchiveTests(unittest.TestCase):
    def test_existing_inference_or_candidate_evidence_cannot_be_overwritten(self):
        for filename in ('inference.json', '候选源图证据.json'):
            with self.subTest(filename=filename), tempfile.TemporaryDirectory() as folder:
                root = Path(folder)
                result = root/filename
                result.write_text('original sealed result')
                task = root/'task.json'
                task.write_text('{}')
                with patch('sys.argv', ['worker', '--task', str(task)]), contextlib.redirect_stderr(io.StringIO()):
                    with self.assertRaises(SystemExit) as error:
                        main()
                self.assertEqual(error.exception.code, 2)
                self.assertEqual(result.read_text(), 'original sealed result')
                self.assertEqual(task.read_text(), '{}')


if __name__ == '__main__':
    unittest.main()
