"""Observe existing first-prediction boundaries; no serving/math changes.

This wrapper adds CPU timestamps to the preserved original-MTP cohort. It does
not emit tokens to a client and is not a primary latency benchmark.
"""
import os
import runpy
import sys
import time
from pathlib import Path

sys.path.insert(0, os.environ['GARNET_RUNTIME_ROOT'])
repo = Path(__file__).resolve().parents[2]
assert (repo / 'runtime/cuda_serving.py').is_file()
sys.path.insert(0, str(repo))
from runtime.cuda_serving import Model

original_infer = Model.infer_batch
request_index = 0


def observed(self, *args, **kwargs):
    global request_index
    index = request_index
    request_index += 1
    assert self.speculative is not None
    began = time.monotonic_ns()
    stamps = {}
    target_predictions = []
    prepared = []
    restore = []

    def stamp(name):
        assert name not in stamps, ('Duplicate first-token boundary', name)
        stamps[name] = time.monotonic_ns() - began

    def patch(obj, name, value):
        restore.append((obj, name, name in obj.__dict__, obj.__dict__.get(name)))
        setattr(obj, name, value)

    original_prepare = self._prepare
    original_chunk = self.decoder.chunk_hidden
    original_sequence = self.decoder.prefill_sequence
    original_prefill = self.speculative.prefill
    original_step = self.speculative._step

    def prepare(*a, **k):
        start = time.monotonic_ns() - began
        value = original_prepare(*a, **k)
        prepared.append({'start_ns': start, 'end_ns': time.monotonic_ns() - began})
        return value

    def chunk(hidden_tokens, active_tokens, positions, predict=True, **k):
        prediction = original_chunk(hidden_tokens, active_tokens, positions, predict, **k)
        if predict:
            stamp('target_prediction_ready')
            target_predictions.append(list(prediction))
        return prediction

    def sequence(*a, **k):
        value = original_sequence(*a, **k)
        stamp('decoder_prefill_returned')
        assert list(value) == target_predictions[0]
        return value

    def step(*a, **k):
        stage = k.get('stage', a[5] if len(a) > 5 else None)
        if stage == 'prefill':
            stamp('initial_draft_started')
        value = original_step(*a, **k)
        if stage == 'prefill':
            stamp('initial_draft_returned')
        return value

    def prefill(*a, **k):
        value = original_prefill(*a, **k)
        stamp('speculative_prefill_returned')
        assert list(value) == target_predictions[0]
        return value

    try:
        patch(self, '_prepare', prepare)
        patch(self.decoder, 'chunk_hidden', chunk)
        patch(self.decoder, 'prefill_sequence', sequence)
        patch(self.speculative, '_step', step)
        patch(self.speculative, 'prefill', prefill)
        result = original_infer(self, *args, **kwargs)
        stamp('complete_answer_returned')
    finally:
        for obj, name, existed, value in reversed(restore):
            if existed:
                setattr(obj, name, value)
            else:
                delattr(obj, name)

    assert len(target_predictions) == 1 and len(prepared) == self.batch
    order = ('target_prediction_ready', 'decoder_prefill_returned',
             'initial_draft_started', 'initial_draft_returned',
             'speculative_prefill_returned', 'complete_answer_returned')
    assert set(stamps) == set(order)
    assert list(stamps[n] for n in order) == sorted(stamps.values())
    for row, prediction in enumerate(target_predictions[0]):
        tokens = result['generated_token_ids'][row]
        first = tokens[0] if tokens else result['stop_token_ids'][row]
        assert prediction == first, ('First target prediction changed', index, row)
    result['first_token_boundary_diagnostic'] = {
        'request_index': index, 'timestamps_ns_from_probe_entry': stamps,
        'first_target_token_ids': target_predictions[0], 'serial_input_rows': prepared,
        'first_target_matches_final_output_or_stop': True,
        'client_streaming_tested': False, 'primary_latency_accepted': False,
        'scope': 'CPU timestamps around the existing original-MTP calls. First target prediction '
                 'is observed before the final MTP prompt hook finishes. The target-to-prefill-return '
                 'gap includes that hook and initial draft/shortlist setup. No GPU synchronization '
                 'or tensor snapshots added; no change to math, native calls or final API-return behavior.'}
    return result


try:
    Model.infer_batch = observed
    entry = Path(__file__).with_name('benchmark_qwen35_shared_mtp.py')
    sys.argv[0] = str(entry)
    sys.argv[3:] = ['baseline']
    runpy.run_path(str(entry), run_name='__main__')
    assert request_index == 28
finally:
    Model.infer_batch = original_infer
