"""CPU fixtures; no actual Garnet, model or profiler imports."""
from pathlib import Path
from types import SimpleNamespace
import ast
import json
import tempfile
import unittest

from .native_cuda_phase_bootstrap_log import NativeCUDAPhaseLoggedTrace
from .test_native_host_stage_trace import RemoveTrace

M = Path(__file__).resolve().parent


class Controls(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='synthetic-bootstrap-')
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / 'results').mkdir()

    def test_original_exception_and_traceback_are_recorded_without_suppression(self):
        for index, error in enumerate((RuntimeError('synthetic native construction failure'), SystemExit(7))):
            path = self.root / 'results' / ('trace' + str(index) + '.jsonl')
            trace = NativeCUDAPhaseLoggedTrace(self.root, 'ordinary', path)
            with self.assertRaises(type(error)):
                try:
                    raise error
                finally:
                    trace.close()
            rows = [json.loads(line) for line in path.with_suffix('.bootstrap.jsonl').read_bytes().splitlines()]
            failure = next(row for row in rows if row['stage'] == 'native_worker_finally_entered')
            self.assertEqual(failure['exception']['type'], type(error).__name__)
            self.assertIn(type(error).__name__, failure['exception']['traceback'])
            self.assertTrue(trace.closed)

    def test_binding_and_ready_are_logged_without_loading_model(self):
        path = self.root / 'results/trace.jsonl'
        trace = NativeCUDAPhaseLoggedTrace(self.root, 'ordinary', path)
        trace.bind_io(SimpleNamespace())
        trace.observe({'event': 'ready', 'generation_mode': 'ordinary'})
        trace.close()
        rows = [json.loads(line) for line in path.with_suffix('.bootstrap.jsonl').read_bytes().splitlines()]
        self.assertEqual([row['stage'] for row in rows], ['trace_constructor_entered', 'trace_constructor_completed',
            'decoder_constructed_and_io_bound', 'native_ready_observed_before_pipe_emit', 'native_worker_finally_entered', 'trace_closed'])
        self.assertIsNone(rows[-2]['exception'])
        with self.assertRaises(FileExistsError):
            NativeCUDAPhaseLoggedTrace(self.root, 'ordinary', path)

    def test_logged_worker_still_has_exact_qualified_model_operations(self):
        candidate = ast.parse((M / 'native_cuda_phase_logged_worker.py').read_bytes())
        for node in candidate.body:
            if isinstance(node, ast.ImportFrom) and node.module.endswith('.native_cuda_phase_bootstrap_log'):
                node.module = node.module.removesuffix('.native_cuda_phase_bootstrap_log') + '.native_host_stage_trace'
                node.names = [ast.alias(name='NativeHostStageTrace')]
        candidate = RemoveTrace().visit(candidate)
        original = ast.parse((M / 'native_fixed_length_stream_worker.py').read_bytes())
        candidate.body = candidate.body[1:]
        original.body = original.body[1:]
        self.assertEqual(ast.dump(candidate, include_attributes=False), ast.dump(original, include_attributes=False))
