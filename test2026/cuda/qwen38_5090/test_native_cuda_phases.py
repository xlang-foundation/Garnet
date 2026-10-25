"""Synthetic CPU clocks/IO/SQLite only; no imported GPU runtime or model result."""
from pathlib import Path
from types import SimpleNamespace
import ast
import copy
import json
import sqlite3
import tempfile
import unittest

from .native_cuda_phase_trace import NativeCUDAPhaseTrace
from .replay_native_cuda_phases import replay, summarize, union_ns
from .test_native_host_stage_trace import RemoveTrace
from .replay_native_host_stage_trace import replay as replay_host

M = Path(__file__).resolve().parent


class Controls(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='synthetic-cuda-phase-')
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / 'results').mkdir()
        self.ticks = 0.0
        self.io_calls = []

    def clock(self):
        self.ticks += 1.0
        return self.ticks

    def trace(self, mode='ordinary'):
        path = self.root / 'results' / (mode + '.jsonl')
        trace = NativeCUDAPhaseTrace(self.root, mode, path, clock=self.clock)
        self.addCleanup(trace.close)
        def from_host(values, shape):
            self.assertEqual(shape, [1, len(values)])
            self.assertTrue(all(v == .125 for v in values))
            self.io_calls.append(shape)
            return object()
        trace.bind_io(SimpleNamespace(from_host=from_host))
        return trace, path

    def fixture(self, mode='ordinary'):
        trace, path = self.trace(mode)
        events = [{'event': 'ready', 'generation_mode': mode, 'batch': 1, 'capacity': 4096,
                   'queued_concurrency': True, 'explicit_ignore_eos_supported': True}]
        trace.observe(events[0])
        trace.start('synthetic-request')
        phases = ['reset', 'input_text_or_image', 'embedding_and_rows']
        suffix = ['mtp_prefill', 'mtp_generate'] if mode == 'mtp' else ['target_prefill', 'target_generate']
        def token(i):
            event = {'event': 'token', 'id': 'synthetic-request', 'sampled_tokens': i + 1,
                     'token_id': i, 'native_monotonic_seconds': self.clock()}
            events.append(event)
            trace.observe(event)
        for name in phases:
            with trace.stage(name):
                pass
        with trace.stage(suffix[0]):
            if mode == 'mtp':
                token(0)
        with trace.stage(suffix[1]):
            for i in range(1 if mode == 'mtp' else 0, 128):
                token(i)
        end = {'event': 'completed', 'id': 'synthetic-request', 'prompt_ids': [10] * 3968,
               'generated_ids': list(range(128)), 'image_sha256': None, 'image_grid': None,
               'ignore_eos': True, 'finish_reason': 'length', 'physical_batch': 1,
               'real_rows': 1, 'padded_rows': 0, 'native_elapsed_seconds': 1.0}
        events.append(end)
        trace.observe(end)
        trace.finish()
        trace.close()
        rows = [json.loads(line) for line in path.read_bytes().splitlines()]
        markers = [json.loads(line) for line in path.with_suffix('.cuda-markers.jsonl').read_bytes().splitlines()]
        expected = {'synthetic-request': {'case': 'text', 'image_sha256': None,
                    'prompt_ids': end['prompt_ids'], 'generated_ids': end['generated_ids']}}
        self.assertEqual(replay_host(rows, events, expected)['status'], 'native_instrumented_host_stages_replayed')
        return markers, rows

    def database(self, markers):
        c = sqlite3.connect(':memory:')
        self.addCleanup(c.close)
        c.executescript('''
            create table StringIds(id integer,value text);
            insert into StringIds values(1,'synthetic kernel'),(2,'synthetic API');
            create table CUPTI_ACTIVITY_KIND_MEMCPY(start integer,end integer,globalPid integer,deviceId integer,contextId integer,streamId integer,bytes integer,copyKind integer,correlationId integer);
            create table CUPTI_ACTIVITY_KIND_KERNEL(start integer,end integer,globalPid integer,deviceId integer,contextId integer,demangledName integer,correlationId integer,graphNodeId integer);
            create table CUPTI_ACTIVITY_KIND_RUNTIME(start integer,end integer,globalTid integer,nameId integer,correlationId integer);
            create table TARGET_INFO_GPU(id integer,name text,uuid text,computeMajor integer,computeMinor integer);
            insert into TARGET_INFO_GPU values(0,'synthetic CPU fixture GPU','synthetic',12,0);
        ''')
        for i, marker in enumerate(markers):
            start = 1_000_000 + i * 1000
            c.execute('insert into CUPTI_ACTIVITY_KIND_MEMCPY values(?,?,?,?,?,?,?,?,?)',
                      (start, start + 10, 1 << 24, 0, 1, 7, marker['expected_htod_bytes'], 1, i))
        for i in range(5):
            start = 1_000_000 + (2 * i + 1) * 1000 + 100
            c.execute('insert into CUPTI_ACTIVITY_KIND_KERNEL values(?,?,?,?,?,?,?,?)',
                      (start, start + 400, 1 << 24, 0, 1, 1, 100 + i, 10 + i))
            c.execute('insert into CUPTI_ACTIVITY_KIND_RUNTIME values(?,?,?,?,?)',
                      (start - 50, start + 100, (1 << 24) + 42, 2, 100 + i))
        return c

    def test_complete_worker_ast_is_original_after_only_marker_trace_removal(self):
        candidate = ast.parse((M / 'native_cuda_phase_stream_worker.py').read_bytes())
        for node in candidate.body:
            if isinstance(node, ast.ImportFrom) and node.module.endswith('.native_cuda_phase_trace'):
                node.module = node.module.removesuffix('.native_cuda_phase_trace') + '.native_host_stage_trace'
                node.names = [ast.alias(name='NativeHostStageTrace')]
        candidate = RemoveTrace().visit(candidate)
        original = ast.parse((M / 'native_fixed_length_stream_worker.py').read_bytes())
        candidate.body = candidate.body[1:]
        original.body = original.body[1:]
        self.assertEqual(ast.dump(candidate, include_attributes=False), ast.dump(original, include_attributes=False))

    def test_both_modes_bind_all_markers_and_unchanged_host_token_replay(self):
        for mode in ('ordinary', 'mtp'):
            markers, rows = self.fixture(mode)
            result = replay(self.database(markers), markers, rows)
            self.assertEqual(result['unique_markers'], 12)
            self.assertEqual(len(result['intervals']), 6)
            self.assertEqual(result['intervals'][1]['kernel']['clipped_interval_union_ns'], 400)
            self.assertEqual(result['intervals'][1]['runtime_api']['records'], 1)
            self.assertFalse(result['performance_admitted'])

    def test_marker_clock_never_translated_to_nsight_epoch(self):
        markers, rows = self.fixture()
        c = self.database(markers)
        initial = replay(c, markers, rows)
        shifted = copy.deepcopy(markers)
        host = copy.deepcopy(rows)
        for marker in shifted:
            marker['host_before'] += 9_000_000_000
            marker['host_after'] += 9_000_000_000
        for row in host:
            for key in ('started', 'completion_observed', 'finished'):
                row[key] += 9_000_000_000
            for phase in row['phases']:
                phase['started'] += 9_000_000_000
                phase['finished'] += 9_000_000_000
        self.assertEqual(initial, replay(c, shifted, host))

    def test_missing_duplicate_wrong_direction_and_wrong_process_markers_rejected(self):
        markers, rows = self.fixture()
        for sql in (
            'delete from CUPTI_ACTIVITY_KIND_MEMCPY where correlationId=3',
            'insert into CUPTI_ACTIVITY_KIND_MEMCPY select * from CUPTI_ACTIVITY_KIND_MEMCPY where correlationId=3',
            'update CUPTI_ACTIVITY_KIND_MEMCPY set copyKind=2 where correlationId=3',
            'update CUPTI_ACTIVITY_KIND_MEMCPY set globalPid=33554432 where correlationId=3',
            'update CUPTI_ACTIVITY_KIND_MEMCPY set streamId=8 where correlationId=3',
            'update CUPTI_ACTIVITY_KIND_MEMCPY set contextId=2 where correlationId=3',
            'update CUPTI_ACTIVITY_KIND_MEMCPY set start=999999 where correlationId=3',
        ):
            with self.subTest(sql=sql):
                c = self.database(markers)
                c.execute(sql)
                with self.assertRaises(AssertionError):
                    replay(c, markers, rows)

    def test_marker_metadata_ownership_count_phase_and_clock_faults_rejected(self):
        markers, rows = self.fixture()
        c = self.database(markers)
        faults = [(0, 'id', 'other'), (1, 'ordinal', True), (2, 'phase', 'other'),
                  (3, 'expected_htod_bytes', 1), (4, 'mode', 'mtp'),
                  (5, 'status', 'failed'), (6, 'host_before', float('nan')),
                  (7, 'host_after', 1e30)]
        for index, key, value in faults:
            damaged = copy.deepcopy(markers)
            damaged[index][key] = value
            with self.subTest(key=key), self.assertRaises(AssertionError):
                replay(c, damaged, rows)
        for damaged in (markers[:-1], markers + markers):
            with self.assertRaises(AssertionError):
                replay(c, damaged, rows)

    def test_missing_gpu_kernel_graph_records_or_process_api_rejected(self):
        markers, rows = self.fixture()
        for sql in (
            'delete from CUPTI_ACTIVITY_KIND_KERNEL',
            'update CUPTI_ACTIVITY_KIND_KERNEL set graphNodeId=0',
            'update TARGET_INFO_GPU set computeMajor=8',
            'update CUPTI_ACTIVITY_KIND_RUNTIME set globalTid=33554433',
        ):
            c = self.database(markers)
            c.execute(sql)
            with self.assertRaises(AssertionError):
                replay(c, markers, rows)

    def test_overlap_union_and_crossing_clipping_do_not_add_gpu_and_api_time(self):
        records = [{'start': 0, 'end': 8, 'name': 'a'}, {'start': 5, 'end': 20, 'name': 'b'},
                   {'start': 30, 'end': 40, 'name': 'a'}]
        result, clipped = summarize(records, 3, 35)
        self.assertEqual(union_ns(clipped), 22)
        self.assertEqual(result['clipped_interval_union_ns'], 22)
        self.assertEqual(sum(r['clipped_sum_duration_ns'] for r in result['by_name']), 25)
        self.assertEqual(sum(r['boundary_crossing_records'] for r in result['by_name']), 2)

    def test_aborted_marker_and_original_error_preserved(self):
        trace, path = self.trace()
        trace.observe({'event': 'ready', 'generation_mode': 'ordinary'})
        trace.start('synthetic-failure')
        with self.assertRaisesRegex(RuntimeError, 'synthetic failure'):
            with trace.stage('target_prefill'):
                raise RuntimeError('synthetic failure')
        trace.close()
        markers = [json.loads(line) for line in path.with_suffix('.cuda-markers.jsonl').read_bytes().splitlines()]
        self.assertEqual(markers[-1]['status'], 'failed')
        self.assertEqual(json.loads(path.read_bytes())['status'], 'instrumented_request_aborted')

    def test_existing_output_is_preserved_and_binding_must_precede_ready(self):
        trace, path = self.trace()
        with self.assertRaises(ValueError):
            trace.bind_io(object())
        trace.close()
        before = path.read_bytes()
        with self.assertRaises(FileExistsError):
            NativeCUDAPhaseTrace(self.root, 'ordinary', path)
        self.assertEqual(path.read_bytes(), before)
        other = self.root / 'results' / 'existing.jsonl'
        other.with_suffix('.cuda-markers.jsonl').write_bytes(b'preserve')
        with self.assertRaises(FileExistsError):
            NativeCUDAPhaseTrace(self.root, 'ordinary', other)
        self.assertFalse(other.exists())

    def test_native_close_witness_requires_completed_trace_and_cannot_be_overwritten(self):
        trace, path = self.trace()
        with self.assertRaises(ValueError):
            trace.model_closed('a' * 64)
        trace.close()
        with self.assertRaises(ValueError):
            trace.model_closed('a' * 64)
        markers, rows = self.fixture('mtp')
        # The actual worker invokes this only after native close calls. The CPU
        # fixture supplies no model and tests only witness ownership/validation.
        witness = NativeCUDAPhaseTrace.__new__(NativeCUDAPhaseTrace)
        witness.closed = True
        witness.current = witness.active = None
        witness.marker_tensors = [object()] * len(markers)
        witness.mode = 'mtp'
        witness.model_closed_path = path.with_name('synthetic-model-close.json')
        witness.model_closed('a' * 64)
        self.assertEqual(json.loads(witness.model_closed_path.read_bytes())['marker_count'], 12)
        with self.assertRaises(FileExistsError):
            witness.model_closed('a' * 64)


if __name__ == '__main__':
    unittest.main()
