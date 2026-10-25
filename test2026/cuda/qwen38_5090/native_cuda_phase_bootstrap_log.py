"""Explicit diagnostic bootstrap log; no CUDA calls or model behavior changes."""
from pathlib import Path
import json
import sys
import traceback

from .native_cuda_phase_trace import NativeCUDAPhaseTrace


class NativeCUDAPhaseLoggedTrace(NativeCUDAPhaseTrace):
    def __init__(self, root, mode, path, **kwargs):
        root, path = Path(root).resolve(), Path(path).resolve()
        if mode not in ('ordinary', 'mtp') or not path.is_relative_to(root / 'results'):
            raise ValueError('Explicit diagnostic mode and owned results path required')
        self.bootstrap_path = path.with_suffix('.bootstrap.jsonl')
        if self.bootstrap_path.exists():
            raise FileExistsError('Preserve original bootstrap log')
        with self.bootstrap_path.open('x', encoding='utf8') as stream:
            stream.write(json.dumps({'stage': 'trace_constructor_entered', 'mode': mode}) + '\n')
        try:
            super().__init__(root, mode, path, **kwargs)
        except BaseException:
            self._bootstrap('trace_constructor_failed', sys.exc_info())
            raise
        self._bootstrap('trace_constructor_completed')

    def _bootstrap(self, stage, exception=None):
        error = None
        if exception is not None and exception[0] is not None:
            error = {'type': exception[0].__name__, 'repr': repr(exception[1]),
                     'traceback': ''.join(traceback.format_exception(*exception))}
        with self.bootstrap_path.open('a', encoding='utf8') as stream:
            stream.write(json.dumps({'stage': stage, 'exception': error}, allow_nan=False) + '\n')

    def bind_io(self, io):
        super().bind_io(io)
        self._bootstrap('decoder_constructed_and_io_bound')

    def observe(self, event):
        super().observe(event)
        if event['event'] == 'ready':
            self._bootstrap('native_ready_observed_before_pipe_emit')

    def close(self):
        if self.closed:
            return
        self._bootstrap('native_worker_finally_entered', sys.exc_info())
        try:
            super().close()
        except BaseException:
            self._bootstrap('trace_close_failed', sys.exc_info())
            raise
        self._bootstrap('trace_closed')

    def model_closed(self, library_sha256):
        super().model_closed(library_sha256)
        self._bootstrap('native_model_close_calls_returned')
