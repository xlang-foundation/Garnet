"""Real-image bitwise vision graph admission; never primary timing."""
import hashlib
import json
import os
import sys

sys.path.insert(0, os.environ['GARNET_RUNTIME_ROOT'])
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import garnet
from qwen35_benchmark_workload import batch_inputs, input_hashes

checkpoint, dataset, output, name = sys.argv[1:5]
assert not os.path.exists(output), 'Preserve vision component evidence'
with open(os.path.join(checkpoint, 'config.json')) as stream:
    config = json.loads(stream.read())
if name == 'Qwen3.5-2B':
    from xModel.qwen35.vl_2b import model as factory
else:
    assert name == 'Qwen3.5-4B'
    from xModel.qwen35.vl_4b import model as factory
io = garnet.cuda_tensor_io()
encoder = None
records = []
result = {'model': name, 'diagnostic_only': True, 'primary_performance_eligible': False,
          'same_compiled_engine_for_both_modes': True, 'records': records}


def save():
    with open(output, 'w') as stream:
        stream.write(json.dumps(result, indent=2))


def digest(value):
    data = io.finite_f32_bytes(value)
    return {'sha256': hashlib.sha256(data).hexdigest(), 'finite_fp32_words': len(data) // 4}


try:
    encoder = factory.create_cuda_vision(checkpoint, config, 12, 20)
    paths = []
    for trial in (2, 3):
        for group in range(4):
            _, selected = batch_inputs(dataset, trial, group, 1, True)
            paths.append(selected[0])
    assert len(set(paths)) == 8
    reference = {}
    last_output = None
    for mode in ('ordinary', 'graph'):
        if mode == 'graph':
            encoder.enable_cuda_graph()
            try:
                encoder.enable_cuda_graph()
                raise AssertionError('Double vision enable was accepted')
            except ValueError:
                pass
        for repeat in range(2):
            for index, path in enumerate(paths):
                prepared = io.prepare_image(path, 65536)
                assert prepared[1] == 12 and prepared[2] == 20
                before = digest(prepared[0])
                if mode == 'graph':
                    io.zero(encoder.input_buffer)
                last_output = encoder.encode_patches(prepared[0])
                actual = digest(last_output)
                assert actual['finite_fp32_words'] == 60 * config['vision_config']['out_hidden_size']
                assert digest(prepared[0]) == before, 'Vision changed source patches'
                if mode == 'graph':
                    assert digest(encoder.input_buffer) == before, 'Incomplete staging refresh'
                if mode == 'ordinary' and repeat == 0:
                    reference[index] = actual
                else:
                    assert actual == reference[index], ('Visual FP32 words differ', mode, repeat, index)
                records.append({'mode': mode, 'repeat': repeat, 'image_index': index,
                                'input_sha256': input_hashes([path]), 'visual_hash': actual,
                                'source_patches_unchanged': True})
                save()
        stats = json.loads(encoder.graph.stats_json())
        result[mode + '_stats'] = stats
        if mode == 'ordinary':
            assert not stats['cuda_graph_requested'] and stats['graph_captures'] == 0
        else:
            assert stats['cuda_graph_requested'] and stats['serial_graph_reuse_requested']
            assert stats['contexts'] == 1 and stats['graph_captures'] == 1 and stats['graph_replays'] > 0
            assert stats['graph_capture_failures'] == 0
            assert encoder.input_buffer_bytes() == 1 * 12 * 20 * 1536 * 4
        save()
    # Close with work pending: captured input/output owners must remain valid.
    prepared = io.prepare_image(paths[0], 65536)
    io.zero(encoder.input_buffer)
    last_output = encoder.encode_patches(prepared[0])
    prepared = None
    encoder.close()
    assert digest(last_output) == reference[0], 'Published vision output lost its owner after pending close'
    for operation in (lambda: encoder.encode_patches(None), lambda: encoder.enable_cuda_graph()):
        try:
            operation()
            raise AssertionError('Closed vision encoder accepted execution')
        except ValueError:
            pass
    result.update({'all_passed': True, 'cases': len(records), 'unique_images': 8,
                   'extra_owned_staging_bytes': 1474560, 'output_survives_close': True,
                   'additional_pending_close_execution': True,
                   'closed_and_double_enable_rejected': True,
                   'scope': 'Same serialized engine,16 ordinary then16 graph executions, all finite visual '
                            'FP32 words and source patches, eight unique real images, changing caller buffers, '
                            'poisoned staging and output lifetime. No independent vision oracle, decoder state, '
                            'full-request latency, warmed process memory or2x acceptance.'})
    save()
    print('VISION_GRAPH_COMPONENT_COMPLETE', name, len(records), flush=True)
except Exception as error:
    result['failure'] = repr(error)
    save()
    raise
finally:
    if encoder is not None:
        encoder.close()
    io.close()
