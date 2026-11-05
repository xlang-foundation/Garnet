"""Evaluator integrity/containment gates; never run alongside inference."""
import json
import socket
from pathlib import Path
import tempfile
import unittest

from evaluate_local_coder_quality import execute, extract_code
from run_local_coder_baseline import engine_processes


class QualityEvaluatorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if engine_processes():
            raise unittest.SkipTest('Do not overlap evaluator processes with reference measurements')

    def test_contract_sources_and_code_extraction(self):
        suite = json.loads(Path(__file__).with_name('local_coder_quality_suite.json').read_text())
        self.assertEqual(len({task['id'] for task in suite['tasks']}), 8)
        for task in suite['tasks']:
            compile(task['tests'], task['id'], 'exec')
        self.assertEqual(extract_code('```python\nx = 1\n```'), 'x = 1')
        with self.assertRaises(ValueError):
            extract_code('```python\nx=1\n```\nExtra prose')

    def test_pass_failure_and_early_exit_are_distinct(self):
        self.assertEqual(execute('value = 42', 'assert value == 42')['status'], 'passed')
        self.assertEqual(execute('value = 41', 'assert value == 42')['status'], 'failed')
        self.assertEqual(execute('raise SystemExit(0)', 'assert False')['status'], 'failed')
        self.assertEqual(execute('import os; os._exit(0)', 'assert False')['status'], 'failed')

    def test_filesystem_and_network_are_contained(self):
        with tempfile.TemporaryDirectory() as temporary, socket.socket() as listener:
            listener.bind(('127.0.0.1', 0))
            listener.listen(1)
            port = listener.getsockname()[1]
            victim = Path(temporary).resolve()/'outside-evaluator.txt'
            victim.write_text('unchanged')
            code = ('import socket\n'
                    'blocked = 0\n'
                    'try:\n'
                    '    open('+repr(str(victim))+", 'w').write('changed')\n"
                    'except OSError:\n'
                    '    blocked += 1\n'
                    'try:\n'
                    '    socket.create_connection(("127.0.0.1", '+str(port)+'), timeout=.5)\n'
                    'except OSError:\n'
                    '    blocked += 1\n')
            result = execute(code, 'assert blocked == 2')
            self.assertEqual(result['status'], 'passed', result)
            self.assertEqual(victim.read_text(), 'unchanged')

    def test_nonterminating_code_is_bounded(self):
        result = execute('while True: pass', 'assert False', seconds=.2)
        self.assertEqual(result['status'], 'timeout', result)


if __name__ == '__main__':
    unittest.main()
