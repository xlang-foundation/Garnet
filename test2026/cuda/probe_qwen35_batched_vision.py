"""Unwired batched-vision screen; run only after other GPU engines exit.

Serial and batched engines are never resident together. Numerical bounds are
unchanged from test_qwen35_linear.py (2e-5 absolute +2e-4 relative). This is not
an independent vision oracle, a full-request admission or primary timing.
"""
import hashlib
import json
import os
import struct
import sys

sys.path.insert(0, os.environ['GARNET_RUNTIME_ROOT'])
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import garnet
from qwen35_benchmark_workload import batch_inputs, input_hashes
from xModel.qwen35.vl_0_8b.cuda_decoder import CUDAImageEncoder

checkpoint, dataset, output, name = sys.argv[1:5]
batch = int(sys.argv[5])
assert batch in (1, 2, 4, 8)
workspace = int(os.environ.get('GARNET_QWEN35_BATCHED_VISION_BUILDER_WORKSPACE_BYTES', '67108864'))
assert workspace in (67108864, 134217728)
assert not os.path.exists(output), 'Preserve batched vision screen'
with open(os.path.join(checkpoint, 'config.json')) as stream:
    config = json.loads(stream.read())
if name == 'Qwen3.5-2B':
    from xModel.qwen35.vl_2b import model as factory
else:
    assert name == 'Qwen3.5-4B'
    from xModel.qwen35.vl_4b import model as factory
factory.validate_config(config)
hidden = config['vision_config']['out_hidden_size']
io = garnet.cuda_tensor_io()
encoder = None
reference = {}
batched_reference = {}
records = []
result = {'model': name, 'vision_batch': batch, 'diagnostic_only': True,
          'serial_builder_workspace_limit_bytes': 67108864, 'batched_builder_workspace_limit_bytes': workspace,
          'primary_performance_eligible': False, 'serial_engine_closed_before_batched_load': False,
          'absolute_tolerance': 2e-5, 'relative_tolerance': 2e-4, 'records': records,
          'independent_vision_quality_accepted': False, 'full_request_accepted': False}


def save():
    with open(output, 'w') as stream:
        stream.write(json.dumps(result, indent=2))


def finite_bytes(value):
    return io.finite_f32_bytes(value)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def compare(actual, expected):
    assert len(actual) == len(expected) == 60 * hidden * 4
    if actual == expected:
        return {'bitwise_equal': True, 'within_original_bound': True, 'max_absolute_error': 0.,
                'max_bound_fraction': 0., 'out_of_bound_words': 0, 'first_outliers': []}
    maximum = fraction = 0.
    failures = 0
    outliers = []
    for index, (av, ev) in enumerate(zip(struct.iter_unpack('<f', actual), struct.iter_unpack('<f', expected))):
        value, reference_value = av[0], ev[0]
        error = abs(value - reference_value)
        bound = 2e-5 + 2e-4 * abs(reference_value)
        maximum = max(maximum, error)
        fraction = max(fraction, error / bound)
        if error > bound:
            failures += 1
            if len(outliers) < 16:
                outliers.append({'word': index, 'actual': value, 'reference': reference_value, 'bound': bound})
    return {'bitwise_equal': False, 'within_original_bound': failures == 0,
            'max_absolute_error': maximum, 'max_bound_fraction': fraction,
            'out_of_bound_words': failures, 'first_outliers': outliers}


try:
    paths = []
    for group in range(8):
        _, selected = batch_inputs(dataset, 2 + group // 4, group % 4, batch, True)
        paths.extend(selected)
        if len(paths) >= 8:
            paths = paths[:8]
            break
    assert len(set(paths)) == 8
    result['inputs'] = [{'path': path, 'sha256': input_hashes([path])} for path in paths]
    encoder = factory.create_cuda_vision(checkpoint, config, 12, 20)
    for repeat in range(2):
        for index, path in enumerate(paths):
            prepared = io.prepare_image(path, 65536)
            assert prepared[1] == 12 and prepared[2] == 20
            value = encoder.encode_patches(prepared[0])
            data = finite_bytes(value)
            if repeat == 0:
                reference[index] = data
            else:
                assert data == reference[index], 'Serial reference changed across repeats'
            records.append({'mode': 'serial', 'repeat': repeat, 'image_index': index, 'visual_sha256': sha(data)})
            save()
    value = None
    prepared = None
    encoder.close()
    encoder = None
    result['serial_engine_closed_before_batched_load'] = True
    save()
    encoder = CUDAImageEncoder(checkpoint, config['vision_config'], 12, 20, batch, workspace)
    assert json.loads(encoder.graph.stats_json())['builder_workspace_limit_bytes'] == workspace
    any_failure = False
    for mode, repeats in (('batched', 2), ('batched_graph', 3)):
        if mode == 'batched_graph':
            if any_failure:
                break
            encoder.enable_cuda_graph()
        for repeat in range(repeats):
            for start in range(0, 8, batch):
                indices = list(range(start, start + batch))
                if repeat % 2:
                    indices.reverse()
                prepared_rows = [io.prepare_image(paths[index], 65536) for index in indices]
                assert all(row[1] == 12 and row[2] == 20 for row in prepared_rows)
                before = [sha(finite_bytes(row[0])) for row in prepared_rows]
                flat = [io.storage_view(row[0], [1, 240 * 1536]) for row in prepared_rows]
                packed = io.storage_view(io.stack_rows(flat), [batch, 240, 1536])
                if mode == 'batched_graph':
                    io.zero(encoder.input_buffer)
                value = encoder.encode_patches(packed)
                values = io.storage_view(value, [batch, 60 * hidden])
                for row, index in enumerate(indices):
                    data = finite_bytes(io.row(values, row))
                    comparison = compare(data, reference[index])
                    any_failure = any_failure or not comparison['within_original_bound']
                    if mode == 'batched' and repeat == 0:
                        batched_reference[index] = data
                    else:
                        assert data == batched_reference[index], 'Batched row/replay output is not stable'
                    records.append({'mode': mode, 'repeat': repeat, 'image_index': index, 'row': row,
                                    'visual_sha256': sha(data), **comparison})
                assert [sha(finite_bytes(row[0])) for row in prepared_rows] == before, 'Batched encoder mutated source patches'
                if mode == 'batched_graph':
                    staged = io.storage_view(encoder.input_buffer, [batch, 240 * 1536])
                    assert [sha(finite_bytes(io.row(staged, row))) for row in range(batch)] == before, 'Batched staging refresh is incomplete'
                save()
        result[mode + '_stats'] = json.loads(encoder.graph.stats_json())
        if mode == 'batched_graph':
            stats = result[mode + '_stats']
            assert stats['contexts'] == stats['graph_captures'] == 1 and stats['graph_replays'] > 0
            assert stats['graph_capture_failures'] == 0
        save()
    result.update({'all_numeric_bounds_passed': not any_failure,
                   'status': 'COMPONENT_PASSED_FULL_REQUEST_STILL_REQUIRED' if not any_failure else 'REJECTED_ORIGINAL_NUMERICAL_BOUND',
                   'scope': 'Unwired existing batch-aware vision lowerer and existing GPU storage_view/stack_rows/row '
                            'operations. Eight distinct image inputs, serial reference repeats, row permutations and '
                            'same-batch graph replay. Original numerical bounds remain unchanged; bitwise differences '
                            'are retained. CPU hashes/synchronization invalidate primary timing. No decoder-state, '
                            'independent vision quality, full-request, process memory or2x acceptance.'})
    save()
    print('BATCHED_VISION_COMPONENT_COMPLETE', name, batch, result['status'], flush=True)
    assert not any_failure, 'Batched vision failed the original numerical parity bound'
except Exception as error:
    result['failure'] = repr(error)
    save()
    raise
finally:
    if encoder is not None:
        encoder.close()
    io.close()
