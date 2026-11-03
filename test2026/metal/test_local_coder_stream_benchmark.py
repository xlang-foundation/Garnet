"""Accounting checks with a synthetic HTTP fixture, never model performance."""
import json
import tempfile
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from local_coder_stream_benchmark import run_request


class StreamingAccountingTests(unittest.TestCase):
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
