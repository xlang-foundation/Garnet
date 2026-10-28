"""Exact compact status/message propagation and no-write error contracts."""
import hashlib
import json
import os
import struct
import sys
sys.path.insert(0, os.environ['GARNET_RUNTIME_ROOT'])
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import garnet
from xModel.qwen35.vl_0_8b.cuda_packed_decoder import CUDAPackedDecoder

fixture = json.load(open(sys.argv[1]))
output, batch = sys.argv[2], int(sys.argv[3])
assert batch in (1, 8) and not os.path.exists(output)
config, checkpoint = fixture['config'], fixture['checkpoint']
result = {'batch': batch, 'all_passed': False, 'checks': [], 'primary_performance_eligible': False}


def save():
    with open(output, 'w') as stream:
        stream.write(json.dumps(result, indent=2))


model = CUDAPackedDecoder(checkpoint, config, batch, 512, 64, 4,
    retain_linear_inputs=True, register_delta_128=True, flat_concat=True,
    shared_output_storage=True, join_decode_input_projections=True,
    shared_activation_storage=True, internal_packed_stores=True, compact_kv_outputs=True)
model.resident_controls = True
model.serial_target_graph_reuse = True
model.enable_cuda_graph(True)
p = model.programs[4]
full_layers = config['layer_types'].count('full_attention')
replay = None


def state_hash():
    h = hashlib.sha256()
    width = p.state_width
    step = min(width, 1048576 // batch)
    for start in range(0, width, step):
        h.update(model.io.finite_f32_bytes(model.io.slice(model.state, start, [batch, min(step, width - start)])))
    return h.hexdigest()


def ids():
    row = fixture['steps'][0]['tokens']
    return [row[i % len(row)] for i in range(batch)]


def prefill():
    return model.prefill_sequence([model.io.embedding(ids()) for _ in range(5)],
                                  [[[i] * 3 for i in range(5)] for _ in range(batch)])


def expect_error(action, text):
    before = state_hash()
    try:
        action()
        raise AssertionError('Expected native error: ' + text)
    except RuntimeError as error:
        message = str(error)
        assert text in message, ('Unexpected native error', text, message)
    assert state_hash() == before, 'Rejected compact operation modified canonical state'
    return message


try:
    prefill()
    original = p.run
    def invalid_gpu_control(inputs):
        active = [0] * (batch * 4)
        active[-1] = 2
        model.io.update(inputs[2], active)
        packed = original(inputs)
        status = model.io.finite_f32_bytes(model.io.slice(packed, p.status_offset, [batch, full_layers]))
        values = [v[0] for v in struct.iter_unpack('<f', status)]
        assert values == [1.0] * (batch * full_layers), 'Invalid GPU flag was not published'
        result['published_invalid_status'] = values
        return packed
    p.run = invalid_gpu_control
    try:
        message = expect_error(lambda: model.chunk_hidden([model.io.embedding(ids())], [[0] * batch],
            [[[slot] * 3] for slot in model.slots], False), 'Compact KV control error')
    finally:
        p.run = original
    assert model.poisoned
    result['checks'].append({'case': 'gpu_active2_status1_no_write_poison', 'message': message})
    try:
        model.step(ids(), [1] * batch, [[slot] * 3 for slot in model.slots])
        raise AssertionError('Poisoned decoder remained usable')
    except ValueError:
        pass
    model.reset()
    assert not model.poisoned
    prefill()
    model.step(ids(), [1] * batch, [[slot] * 3 for slot in model.slots])
    # A status error only in the last row/last full layer must be consumed.
    data = [0.0] * (batch * p.output_width)
    data[(batch - 1) * p.output_width + p.status_offset + full_layers - 1] = 1.0
    bad = model.io.from_host(data, [batch, p.output_width])
    data = None
    message = expect_error(lambda: p.state_commit.commit_all(bad, model.state, model.slot_tensor, model.active),
                           'Compact KV control error')
    result['checks'].append({'case': 'last_row_status_blocks_copy_commit', 'message': message})
    replay = garnet.cuda_packed_prefix()
    replay.build(checkpoint, json.dumps(p.commit_spec(False)))
    spans = []
    for slot in model.slots:
        spans.extend([slot, 0])
    message = expect_error(lambda: replay.commit(bad, model.state, spans), 'Compact KV control error')
    result['checks'].append({'case': 'last_row_status_blocks_zero_prefix_replay', 'message': message})
    duplicate = model.io.from_host([float(model.slots[row]) for row in range(batch) for _ in range(4)], [batch, 4])
    active = model.io.from_host([1.0 if t < 2 else 0.0 for _ in range(batch) for t in range(4)], [batch, 4])
    message = expect_error(lambda: p.state_commit.commit_all(model.last_packed, model.state, duplicate, active),
                           'unique and in range')
    result['checks'].append({'case': 'duplicate_active_slot_no_write', 'message': message})
    assert all(g['graph_capture_failures'] == 0 for g in model.memory_info()['graphs'])
    result['all_passed'] = True
    result['scope'] = 'Exact GPU pack status, native error messages, nonzero canonical-state preservation, poison/block/reset, last-row status and duplicate-slot guards. No independent model quality or performance; readonly/cross-device contracts remain separate.'
    save()
    print('QWEN35_COMPACT_CONTROL_ERRORS_PASS', batch, flush=True)
except Exception as error:
    result['failure'] = repr(error)
    save()
    raise
finally:
    if replay is not None:
        replay.close()
    model.close()
