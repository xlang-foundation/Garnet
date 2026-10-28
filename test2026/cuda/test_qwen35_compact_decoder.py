"""Sequential original/compact fixture decoders, complete finite state parity."""
import hashlib
import json
import os
import sys

sys.path.insert(0, os.environ['GARNET_RUNTIME_ROOT'])
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import garnet
from xModel.qwen35.vl_0_8b.cuda_packed_decoder import CUDAPackedDecoder

fixture = json.load(open(sys.argv[1]))
output = sys.argv[2]
batch, capacity, internal = int(sys.argv[3]), int(sys.argv[4]), sys.argv[5] == '1'
assert batch in (1, 2, 4, 8) and capacity in (384, 512)
assert not os.path.exists(output), 'Preserve integrated fixture evidence'
config, checkpoint = fixture['config'], fixture['checkpoint']
result = {'batch': batch, 'capacity': capacity, 'internal_stores': internal, 'all_passed': False,
          'independent_model_quality_accepted': False, 'primary_performance_eligible': False,
          'runs': []}


def save():
    with open(output, 'w') as stream:
        stream.write(json.dumps(result, indent=2))


def digest(io, value, width):
    step = min(width, 1048576 // batch)
    value_hash = hashlib.sha256()
    for start in range(0, width, step):
        part = io.slice(value, start, [batch, min(step, width - start)])
        value_hash.update(io.finite_f32_bytes(part))
    return value_hash.hexdigest()


def tokens(index):
    values = fixture['steps'][index % len(fixture['steps'])]['tokens']
    return [values[row % len(values)] for row in range(batch)]


def run(compact):
    record = {'compact': compact, 'snapshots': [], 'negative_checks': []}
    result['runs'].append(record)
    save()
    model = CUDAPackedDecoder(checkpoint, config, batch, capacity, 64, 4,
        register_delta_128=True, flat_concat=True, shared_output_storage=True,
        join_decode_input_projections=True, shared_activation_storage=True,
        internal_packed_stores=internal, compact_kv_outputs=compact)
    model.resident_controls = True
    model.serial_target_graph_reuse = True
    model.enable_cuda_graph(True)
    width = model.programs[4].state_width

    def snapshot(phase, prediction):
        row = {'phase': phase, 'predictions': list(prediction), 'slots': list(model.slots),
               'state_sha256': digest(model.io, model.state, width),
               'last_hidden_sha256': hashlib.sha256(model.io.finite_f32_bytes(model.last_hidden)).hexdigest(),
               'decode_hidden_sha256': hashlib.sha256(model.io.finite_f32_bytes(model.requests[4][0])).hexdigest(),
               'prefill_hidden_sha256': hashlib.sha256(model.io.finite_f32_bytes(model.requests[64][0])).hexdigest()}
        record['snapshots'].append(row)
        save()

    def prefill():
        prompt = [model.io.embedding(tokens(i)) for i in range(7)]
        return model.prefill_sequence(prompt, [[[i] * 3 for i in range(7)] for _ in range(batch)],
                                      [7 - row % 3 for row in range(batch)])

    try:
        if compact:
            program = model.programs[4]
            original_run = program.run
            initial_state = digest(model.io, model.state, width)
            def damaged_controls(inputs):
                values = [0] * (batch * 4)
                values[0] = 2
                model.io.update(inputs[2], values)
                return original_run(inputs)
            program.run = damaged_controls
            try:
                model.chunk_hidden([model.io.embedding(tokens(0))], [[0] * batch], [[[0, 0, 0]] for _ in range(batch)], False)
                raise AssertionError('Invalid GPU control flag reached canonical commit')
            except RuntimeError:
                pass
            finally:
                program.run = original_run
            assert model.poisoned and initial_state == digest(model.io, model.state, width)
            try:
                model.step(tokens(0), [1] * batch, [[0, 0, 0] for _ in range(batch)])
                raise AssertionError('Poisoned decoder remained usable')
            except ValueError:
                pass
            model.reset()
            assert not model.poisoned and initial_state == digest(model.io, model.state, width)
            record['negative_checks'].append('gpu_invalid_control_blocks_commit_and_requires_reset')
        snapshot('prefill', prefill())
        flags = [[int((t + row) % 3 != 1 and row % 4 != 3) for row in range(batch)] for t in range(4)]
        snapshot('inactive_gaps', model.chunk_hidden([model.io.embedding(tokens(t)) for t in range(4)], flags,
            [[[model.slots[row] + t] * 3 for t in range(4)] for row in range(batch)], True))
        for step in range(8):
            snapshot('step' + str(step), model.step(tokens(step), [int((step + row) % 4 != 2) for row in range(batch)],
                                                   [[p] * 3 for p in model.slots]))
        model.reset()
        snapshot('reset_prefill', prefill())
        first, repeated = record['snapshots'][0], record['snapshots'][-1]
        assert all(first[key] == repeated[key] for key in first if key != 'phase'), 'Reset did not restore the original request state'
        model.slots = [capacity - 4] * batch
        boundary_flags = [[int(row == 0 or (t + row) % 3 != 1) for row in range(batch)] for t in range(4)]
        snapshot('capacity_tail', model.chunk_hidden([model.io.embedding(tokens(t)) for t in range(4)], boundary_flags,
            [[[capacity - 4 + t] * 3 for t in range(4)] for _ in range(batch)], True))
        snapshot('inactive_at_capacity', model.step(tokens(0), [0] * batch, [[p] * 3 for p in model.slots]))
        before = digest(model.io, model.state, width)
        try:
            model.step(tokens(0), [1] + [0] * (batch - 1), [[p] * 3 for p in model.slots])
            raise AssertionError('Past-capacity request was accepted')
        except ValueError:
            pass
        assert before == digest(model.io, model.state, width)
        record['negative_checks'].append('past_capacity_rejected_without_state_write')
        if compact:
            program = model.programs[4]
            for field in ('state_width', 'packed_width', 'output_state_width', 'status_offset'):
                specification = json.loads(json.dumps(program.commit_spec(True)))
                specification[field] += 1
                helper = garnet.cuda_packed_prefix()
                try:
                    helper.build(checkpoint, json.dumps(specification))
                    raise AssertionError('Malformed compact schema accepted: ' + field)
                except RuntimeError:
                    record['negative_checks'].append('schema_' + field)
                finally:
                    helper.close()
            specification = json.loads(json.dumps(program.commit_spec(True)))
            specification['layout'][0]['output_offset'] += 1
            helper = garnet.cuda_packed_prefix()
            try:
                helper.build(checkpoint, json.dumps(specification))
                raise AssertionError('Overlapping compact state schema accepted')
            except RuntimeError:
                record['negative_checks'].append('schema_output_interval')
            finally:
                helper.close()
            try:
                alias = model.io.storage_view(model.state, [batch, program.output_width])
                program.state_commit.commit_all(alias, model.state, model.slot_tensor, model.active)
                raise AssertionError('Canonical alias was accepted')
            except RuntimeError:
                record['negative_checks'].append('canonical_alias_rejected')
            assert before == digest(model.io, model.state, width)
        record['owned_memory'] = model.memory_info()
        # Enqueue without a prediction/readback, then close while caller retains
        # state/hidden tensors. Provider fences must preserve those outputs.
        model.chunk_hidden([model.io.embedding(tokens(1))], [[0] * batch], [[[p] * 3] for p in model.slots], False)
        retained_state, retained_hidden = model.state, model.last_hidden
        model.close()
        io = garnet.cuda_tensor_io()
        try:
            record['closed_state_sha256'] = digest(io, retained_state, width)
            record['closed_hidden_sha256'] = hashlib.sha256(io.finite_f32_bytes(retained_hidden)).hexdigest()
        finally:
            io.close()
        assert record['closed_state_sha256'] == before, 'Pending close changed canonical state'
        save()
        return record
    finally:
        model.close()


try:
    control = run(False)
    candidate = run(True)
    assert control['snapshots'] == candidate['snapshots'], 'Compact full-state/hidden/prediction mismatch'
    assert control['closed_state_sha256'] == candidate['closed_state_sha256']
    assert control['closed_hidden_sha256'] == candidate['closed_hidden_sha256']
    assert control['owned_memory']['projection_weight_pool_bytes'] == candidate['owned_memory']['projection_weight_pool_bytes']
    assert candidate['owned_memory']['compact_state_commit_bytes'] > 0
    assert candidate['owned_memory']['engine_output_bytes'] < control['owned_memory']['engine_output_bytes']
    result['all_passed'] = True
    result['scope'] = 'Separate sequential fixture models. Complete finite canonical/hidden bytes and predictions, gaps, reset, capacity tails and retained-output close checks. Not pretrained quality, primary latency or process memory.'
    save()
    print('QWEN35_COMPACT_DECODER_PASS', batch, capacity, internal, flush=True)
except Exception as error:
    result['failure'] = repr(error)
    save()
    raise
