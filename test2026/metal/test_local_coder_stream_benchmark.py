"""Accounting checks with a synthetic HTTP fixture, never model performance."""
import json
import hashlib
import socket
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from local_coder_stream_benchmark import run_request
from run_local_coder_baseline import engine_processes
from summarize_local_coder_baseline import audit, paging


class StreamingAccountingTests(unittest.TestCase):
    def test_campaign_collects_and_closes_two_fresh_fixture_processes(self):
        if engine_processes():
            self.skipTest('Never overlap the fixture with a live inference engine')
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            server = root/'fixture.py'
            server.write_text('''import json,sys,time
from http.server import BaseHTTPRequestHandler,HTTPServer
class Handler(BaseHTTPRequestHandler):
 def log_message(self,*args): pass
 def do_GET(self): self.send_response(200);self.end_headers()
 def do_POST(self):
  self.rfile.read(int(self.headers['Content-Length']))
  self.send_response(200);self.send_header('Content-Type','text/event-stream');self.end_headers()
  for n in (1,3):
   d={'choices':[{'index':0,'delta':{'content':'x'},'finish_reason':'length' if n==3 else None}],'usage':{'prompt_tokens':10,'completion_tokens':n}}
   self.wfile.write(('data: '+json.dumps(d)+'\\n\\n').encode());self.wfile.flush();time.sleep(.01)
  self.wfile.write(b'data: [DONE]\\n\\n');self.wfile.flush()
HTTPServer(('127.0.0.1',int(sys.argv[1])),Handler).serve_forever()
''')
            with socket.socket() as sock:
                sock.bind(('127.0.0.1', 0))
                port = sock.getsockname()[1]
            cases = root/'cases.json'
            cases.write_text(json.dumps([dict(model='fixture', stream=True, max_tokens=3,
                                             stream_options=dict(include_usage=True))]))
            sha = hashlib.sha256(cases.read_bytes()).hexdigest()
            (root/'workload.json').write_text(json.dumps(dict(cases_sha256=sha,
                                                            requests=[dict(prompt_tokens=10)])))
            manifest = dict(scope='Synthetic fixture only', required_fresh_process_cohorts=2,
                            warmup_waves_per_cell=1, measured_waves_per_cell=1,
                            plans=[dict(id='fixture', cases='cases.json', workload='workload.json',
                                        cases_sha256=sha)])
            (root/'manifest.json').write_text(json.dumps(manifest))
            spec = dict(engine='synthetic_fixture_not_model', url='http://127.0.0.1:'+str(port),
                        argv=[sys.executable, str(server), str(port)], startup_observation_seconds=0)
            (root/'engine.json').write_text(json.dumps(spec))
            command = [sys.executable, str(Path(__file__).with_name('run_local_coder_baseline.py')),
                       '--experiment-root', str(root), '--manifest', str(root/'manifest.json'),
                       '--engine-spec', str(root/'engine.json'), '--output', str(root/'out'),
                       '--cohorts', '2']
            result = subprocess.run(command, capture_output=True, text=True, timeout=45)
            self.assertEqual(result.returncode, 0, result.stdout+result.stderr)
            state = json.loads((root/'out/status.json').read_text())
            self.assertEqual(state['completed_cohorts'], 2)
            cohorts = [json.loads((root/'out'/('cohort-'+str(i))/'cohort.json').read_text())
                       for i in (1, 2)]
            self.assertNotEqual(cohorts[0]['pid'], cohorts[1]['pid'])
            self.assertTrue(all(c['status'] == 'completed' and not c['shutdown']['forced_kill']
                                for c in cohorts))
            self.assertTrue(all(c['final_memory_sample']['status'] == 'bracketed' for c in cohorts))
            for i in (1, 2):
                measured = root/'out'/('cohort-'+str(i))/'fixture-measured-0'
                self.assertEqual(json.loads((measured/'summary.json').read_text())['status'], 'completed')
                self.assertTrue((measured/'request-0.events.jsonl').read_text())
            report = audit(root/'out')
            self.assertFalse(any('Missing' in issue or 'Campaign' in issue for issue in report['issues']))
            self.assertEqual(len(report['cohorts']), 2)
            for cohort in report['cohorts']:
                self.assertEqual(len(cohort['cells'][0]['waves']), 1)
                self.assertEqual(cohort['cells'][0]['waves'][0]['generated_tokens'], [3])
            (root/'out/cohort-2/fixture-measured-0/summary.json').unlink()
            self.assertTrue(any('Missing measured wave 0' in issue for issue in audit(root/'out')['issues']))

    def test_paging_counters_preserve_swap_and_compression(self):
        rows = [dict(vm_stat='Pageins: 10.\nSwapouts: 3.\nCompressions: 4.\n'),
                dict(vm_stat='Pageins: 15.\nSwapouts: 9.\nCompressions: 7.\n')]
        delta = paging(rows)
        self.assertEqual(delta['Pageins'], 5)
        self.assertEqual(delta['Swapouts'], 6)
        self.assertEqual(delta['Compressions'], 3)
        self.assertIsNone(delta['Swapins'])

    def test_incremental_final_only_and_error_responses(self):
        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
                mode = body['model']
                if mode == 'error':
                    self.send_response(503)
                    self.end_headers()
                    self.wfile.write(b'fixture unavailable')
                    return
                self.send_response(200)
                self.send_header('Content-Type', 'text/event-stream')
                self.end_headers()
                counts = [1, 3] if mode == 'incremental' else [None, 3]
                for i, count in enumerate(counts):
                    chunk = dict(choices=[dict(index=0, delta=dict(content='x'),
                                              finish_reason='length' if i else None)])
                    if count is not None:
                        chunk['usage'] = dict(prompt_tokens=10, completion_tokens=count)
                    try:
                        self.wfile.write(('data: '+json.dumps(chunk)+'\n\n').encode())
                        self.wfile.flush()
                    except (BrokenPipeError, ConnectionResetError):
                        return  # The rejection tests deliberately close early.
                    time.sleep(.01)
                try:
                    self.wfile.write(b'data: [DONE]\n\n')
                    self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError):
                    return

        server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with tempfile.TemporaryDirectory() as temp:
                url = 'http://127.0.0.1:'+str(server.server_port)
                results = {}
                for mode in ('incremental', 'final-only', 'error'):
                    results[mode] = run_request(url, dict(model=mode), Path(temp)/mode,
                                                threading.Barrier(1), 5)
                incremental = results['incremental']
                self.assertEqual(incremental['generated_tokens'], 3)
                self.assertEqual(incremental['first_usage_batch_tokens'], 1)
                self.assertGreater(incremental['decode_tokens_per_second'], 0)
                self.assertIsNone(results['final-only']['decode_tokens_per_second'])
                self.assertGreater(results['final-only']['ttft_content_seconds'], 0)
                self.assertEqual(results['error']['status'], 'failed')
                self.assertEqual(results['error']['http_status'], 503)
                self.assertEqual(results['error']['http_body'], 'fixture unavailable')
                mismatch = run_request(url, dict(model='incremental'), Path(temp)/'wrong-prompt',
                                       threading.Barrier(1), 5, expected_prompt=11)
                self.assertEqual(mismatch['status'], 'failed')
                self.assertIn('prompt count differs', mismatch['error'])
                over = run_request(url, dict(model='incremental', max_tokens=2), Path(temp)/'over-budget',
                                   threading.Barrier(1), 5)
                self.assertEqual(over['status'], 'failed')
                self.assertIn('exceeded output budget', over['error'])
        finally:
            server.shutdown()
            server.server_close()
            thread.join()


if __name__ == '__main__':
    unittest.main()
