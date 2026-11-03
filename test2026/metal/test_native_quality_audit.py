"""Reject plausible native-quality results when input/output evidence differs."""
import json
from pathlib import Path
import tempfile
import unittest
from audit_local_coder_native_quality import audit, digest


class NativeQualityAuditTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        frozen = self.root/'frozen/task'
        frozen.mkdir(parents=True)
        (self.root/'task').mkdir()
        self.write(frozen/'cases.json', [dict(max_tokens=4)])
        checksum = digest(frozen/'cases.json')
        self.write(frozen/'workload.json', dict(cases_sha256=checksum, requests=[dict(token_ids=[3, 4])]))
        self.write(self.root/'frozen/manifest.json', dict(plans=[dict(id='task', cases_sha256=checksum)]))
        self.write(self.root/'status.json', dict(status='collected_pending_sandbox_evaluation', tasks=[dict(id='task', client_exit_code=0)]))
        self.write(self.root/'task/request-0.result.json', dict(status='completed', cached_prompt_tokens=0,
                   prompt_tokens=2, generated_tokens=1, text='x', finish_reason='stop'))
        self.events = [dict(event='ready', active_requests=1, prefix_cache=False),
                       dict(event='token', id='a', token_id=7, sampled_tokens=1),
                       dict(event='completed', id='a', prompt_ids=[3, 4], generated_ids=[7], text='x', finish_reason='stop')]

    def tearDown(self):
        self.temporary.cleanup()

    def write(self, path, value):
        path.write_text(json.dumps(value))

    def run_audit(self):
        log = self.root/'worker.jsonl'
        log.write_text('\n'.join(json.dumps(dict(line=json.dumps(event))) for event in self.events)+'\n')
        return audit(self.root, log)

    def test_exact_evidence_passes(self):
        self.assertEqual(self.run_audit()['issues'], [])

    def test_changed_prompt_rejected(self):
        self.events[-1]['prompt_ids'] = [3, 5]
        self.assertIn('task: native prompt IDs differ', self.run_audit()['issues'])

    def test_changed_stream_rejected(self):
        self.events[1]['token_id'] = 8
        self.assertTrue(any('streamed/final' in issue for issue in self.run_audit()['issues']))

    def test_missing_completion_rejected(self):
        self.events.pop()
        self.assertIn('Unterminated native request', self.run_audit()['issues'])

    def test_http_text_rejected(self):
        self.events[-1]['text'] = 'different'
        self.assertIn('task: client/native text or finish reason differs', self.run_audit()['issues'])


if __name__ == '__main__':
    unittest.main()
