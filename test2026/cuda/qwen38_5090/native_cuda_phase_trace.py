"""Explicit CUDA diagnostic markers; model operations and serving defaults unchanged.

Each marker owns a separate float32 tensor. Its distinct odd word count identifies
one HtoD copy in Nsight SQLite. Only actual unique copies admit a trace interval;
Python monotonic timestamps are never compared with the Nsight clock.
"""
from contextlib import contextmanager
from pathlib import Path
import json
import re

from .native_host_stage_trace import NativeHostStageTrace

BASE_WORDS = 16381
MAX_MARKERS = 512


class NativeCUDAPhaseTrace(NativeHostStageTrace):
    def __init__(self, root, mode, path, **kwargs):
        marker_path = Path(path).with_suffix('.cuda-markers.jsonl')
        self.model_closed_path = Path(path).with_suffix('.model-closed.json')
        # Preserve either existing output before opening its sibling.
        if marker_path.exists() or Path(path).exists() or self.model_closed_path.exists():
            raise FileExistsError('Preserve prior CUDA diagnostic outputs')
        super().__init__(root, mode, path, **kwargs)
        self.marker_stream = marker_path.open('x', encoding='utf8')
        self.io = None
        self.marker_tensors = []

    def bind_io(self, io):
        if self.closed or self.io is not None or self.current is not None or self.ready is not None:
            raise ValueError('Bind diagnostic IO exactly once before worker ready')
        self.io = io

    def _mark(self, boundary, phase=None, status='completed'):
        if self.closed or self.io is None or self.current is None:
            raise ValueError('Active request and bound diagnostic IO required')
        ordinal = len(self.marker_tensors)
        if ordinal >= MAX_MARKERS:
            raise ValueError('Bounded diagnostic marker budget exceeded')
        words = BASE_WORDS + 2 * ordinal
        before = self.clock()
        tensor = self.io.from_host([.125] * words, [1, words])
        self.marker_tensors.append(tensor)
        after = self.clock()
        row = {'ordinal': ordinal, 'id': self.current['id'], 'mode': self.mode,
               'boundary': boundary, 'phase': phase, 'status': status,
               'float32_words': words, 'expected_htod_bytes': words * 4,
               'host_before': before, 'host_after': after,
               'scope': 'Owned diagnostic HtoD marker, not model data or benchmark timing'}
        self.marker_stream.write(json.dumps(row, allow_nan=False) + '\n')
        self.marker_stream.flush()

    def start(self, key):
        if self.io is None:
            raise ValueError('Bind marker IO before starting a diagnostic request')
        super().start(key)
        self._mark('request_start')

    @contextmanager
    def stage(self, name):
        with super().stage(name):
            self._mark('phase_start', name)
            status = 'failed'
            try:
                yield
                status = 'completed'
            finally:
                self._mark('phase_end', name, status)

    def finish(self):
        if self.current is None or self.active is not None or not self.current['completed']:
            raise ValueError('Completed request and closed phases required')
        self._mark('request_end')
        super().finish()

    def close(self):
        if self.closed:
            return
        super().close()
        self.marker_stream.close()

    def model_closed(self, library_sha256):
        """Called only after all three native model close calls return normally.

        Nsight owns the wrapper exit code. This separate witness records model
        cleanup completion, without treating wrapper success as the child exit.
        """
        if not self.closed or self.current is not None or self.active is not None:
            raise ValueError('Close completed trace and all native model resources first')
        if not self.marker_tensors or re.fullmatch('[0-9a-f]{64}', library_sha256) is None:
            raise ValueError('Actual diagnostic markers and library identity required')
        with self.model_closed_path.open('x', encoding='utf8') as stream:
            json.dump({'status': 'native_model_close_calls_returned', 'mode': self.mode,
                       'marker_count': len(self.marker_tensors), 'library_sha256': library_sha256,
                       'scope': 'Witness after native session/vision/decoder cleanup calls; distinct from Nsight wrapper and interpreter exit'}, stream, indent=2)
