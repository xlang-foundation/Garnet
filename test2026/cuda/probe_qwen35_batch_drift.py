"""Unoptimized 4B identical-prefix diagnostic; serialize B1 then B2 externally.

Extra normalization, full logits and CPU copies invalidate all timing. This
does not change model arithmetic or claim to identify a cause by itself.
"""
import hashlib
import heapq
import json
import math
import os
import struct
import sys
import time
from pathlib import Path

sys.path.insert(0, os.environ['GARNET_RUNTIME_ROOT'])
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import garnet
from runtime.cuda_serving import Model

source_path, output_path = map(Path, sys.argv[1:3])
assert not output_path.exists(), 'Preserve drift diagnostic'
source = json.loads(source_path.read_text())
assert source['all_passed']
request = source['request']
profile = source['ordinary_profile']
batch = profile['batch']
assert request['model_id'] == 'Qwen3.5-4B' and batch in (1, 2)
assert profile['speculativeMTP'] == 0 and not profile['compactKVOutputs']
case = next(item for item in request['cases'] if item['style'] == 'uniform')
expected = next(item['result'] for item in source['ordinary'] if item['style'] == 'uniform')
assert len(set(case['input_sha256'])) == len(set(case['prompts'])) == 1
assert [hashlib.sha256(Path(path).read_bytes()).hexdigest() for path in case['images']] == case['input_sha256']
limit = 27
expected_ids = [tokens[:limit] for tokens in expected['generated_token_ids']]
assert all(len(tokens) == limit for tokens in expected_ids)
evidence = {'source': str(source_path), 'source_sha256': hashlib.sha256(source_path.read_bytes()).hexdigest(),
    'batch': batch, 'profile': profile, 'image_sha256': case['input_sha256'], 'trace': [], 'events': [],
    'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
    'diagnostic_complete': False, 'root_cause_resolved': False,
    'scope': 'Ordinary non-MTP target only. Two reset requests; second captures identical-prefix hidden rows, '
             'actual prefill input hashes and same-query M1/M2 BF16 head logits plus sampled FP64 scores. '
             'Not inference latency, independent quality, a production change or an automatic drift acceptance.'}


def flat(value):
    if isinstance(value, list):
        return [item for part in value for item in flat(part)]
    return [float(value)]


def cpu(value):
    return flat(garnet.tensor_to_cpu(value).tolist())


def save():
    output_path.write_text(json.dumps(evidence, indent=2))


def event(name):
    evidence['events'].append({'event': name, 'unix_seconds': time.time()})
    save()
    print('BATCH_DRIFT_EVENT', batch, name, flush=True)


def bf16(value):
    word = struct.unpack('<I', struct.pack('<f', value))[0]
    word = ((word + 0x7fff + ((word >> 16) & 1)) >> 16) & 65535
    return struct.unpack('<f', struct.pack('<I', word << 16))[0]


event('load_begin')
model = Model(request['checkpoint'], profile, request['model_id'])
try:
    event('load_complete')
    assert not model.decoder.compact_kv_outputs and model.speculative is None
    control = model.infer_batch(case['prompts'], case['images'], limit)
    evidence['control'] = control
    save()
    assert control['generated_token_ids'] == expected_ids, 'Fresh ordinary build differs from the saved prefix'
    event('uninstrumented_control_complete')
    decoder = model.decoder
    width = decoder.config['hidden_size']
    vocab = decoder.config['vocab_size']
    original_prefill = decoder.prefill_sequence
    original_chunk = decoder.chunk_hidden

    def traced_prefill(hidden, positions, lengths=None):
        digests = [hashlib.sha256() for _ in range(batch)]
        for value in hidden:
            data = decoder.io.finite_f32_bytes(value)
            assert len(data) == batch * width * 4
            for row in range(batch):
                digests[row].update(data[row * width * 4:(row + 1) * width * 4])
        evidence['actual_prefill_inputs'] = {'fp32_row_sha256': [item.hexdigest() for item in digests],
            'tokens': len(hidden), 'words_per_row': len(hidden) * width,
            'positions': positions, 'lengths': lengths,
            'hash_order': 'Time then hidden channel, complete finite FP32 input rows'}
        save()
        return original_prefill(hidden, positions, lengths)

    def logits(value, rows):
        scores = cpu(decoder.io.logits(value))
        assert len(scores) == rows * vocab and all(math.isfinite(score) for score in scores)
        summaries, score_rows = [], []
        for row in range(rows):
            values = scores[row * vocab:(row + 1) * vocab]
            score_rows.append(values)
            ids = heapq.nlargest(8, range(vocab), key=lambda token: (values[token], -token))
            summaries.append({'top8': [{'id': token, 'score': values[token]} for token in ids],
                              'watched': {str(token): values[token] for token in (22152, 21861)}})
        return summaries, score_rows

    def traced_chunk(*args, **kwargs):
        prediction = original_chunk(*args, **kwargs)
        predict = args[3] if len(args) > 3 else kwargs.get('predict', True)
        if not predict:
            return prediction
        index = len(evidence['trace'])
        if index >= limit or list(prediction) != [tokens[index] for tokens in expected_ids]:
            evidence['unexpected_prediction'] = {'index': index, 'actual': list(prediction),
                'expected': [tokens[index] for tokens in expected_ids] if index < limit else None}
            save()
            raise AssertionError('Traced ordinary request changed the saved prefix')
        raw = cpu(decoder.last_hidden)
        normalized_tensor = decoder.head.run_device([decoder.last_hidden])
        normalized = cpu(normalized_tensor)
        assert len(raw) == len(normalized) == batch * width
        assert all(math.isfinite(value) for value in raw + normalized)
        item = {'prediction_index_zero_based': index, 'prediction': list(prediction),
                'raw_hidden_rows': [raw[row * width:(row + 1) * width] for row in range(batch)],
                'normalized_hidden_rows': [normalized[row * width:(row + 1) * width] for row in range(batch)]}
        evidence['trace'].append(item)
        if index == 26:
            actual, actual_scores = logits(normalized_tensor, batch)
            evidence['head_isolation'] = {'original_batch': actual}
            save()
            assert actual[0]['top8'][0]['id'] == prediction[0], 'Extra head call changed the current decision'
            single = decoder.io.row(normalized_tensor, 0)
            duplicate = decoder.io.stack_rows([single, single])
            one, one_scores = logits(single, 1)
            two, two_scores = logits(duplicate, 2)
            duplicate_equal = two_scores[0] == two_scores[1]
            selected = sorted({22152, 21861} | {row['id'] for group in (actual[0], one[0], two[0]) for row in group['top8']})
            table = cpu(decoder.io.embedding(selected))
            assert len(table) == len(selected) * width and all(math.isfinite(value) for value in table)
            query = [bf16(value) for value in normalized[:width]]
            unit = 2.0 ** -24
            gamma = width * unit / (1 - width * unit)
            reference = []
            for offset, token in enumerate(selected):
                weight = table[offset * width:(offset + 1) * width]
                assert all(bf16(value) == value for value in weight)
                terms = [a * b for a, b in zip(query, weight)]
                dot, absolute = math.fsum(terms), math.fsum(abs(term) for term in terms)
                bound = gamma * absolute + (abs(dot) + gamma * absolute) / 256.0 + 2.0 ** -133
                observed = {'original_batch_row0': actual_scores[0][token],
                            'same_query_m1': one_scores[0][token], 'same_query_m2_row0': two_scores[0][token]}
                reference.append({'token': token, 'fp64_dot': dot, 'sum_abs_products': absolute,
                                  'original_fixed_fp32_accum_bf16_output_bound': bound,
                                  'fp32_then_bf16_rounded_fp64_dot': bf16(dot), 'weight_fp32_words': weight,
                                  'observed_scores': observed,
                                  'original_fixed_bound_fractions': {name: abs(value - dot) / bound for name, value in observed.items()}})
            evidence['head_isolation'] = {'original_batch': actual, 'same_query_m1': one, 'same_query_m2': two,
                'normalized_row0': normalized[:width], 'bf16_query_row0': query, 'fp64_sampled_rows': reference,
                'duplicate_m2_full_vocab_score_values_equal': duplicate_equal,
                'all_vocab_head_mathematical_argmax_proven': False}
            save()
            assert duplicate_equal, 'Identical head inputs differ inside the same M2 call'
            assert all(fraction <= 1 for row in reference for fraction in row['original_fixed_bound_fractions'].values()), 'Original sampled FP64 head bound failed'
        save()
        return prediction

    decoder.prefill_sequence = traced_prefill
    decoder.chunk_hidden = traced_chunk
    result = model.infer_batch(case['prompts'], case['images'], limit)
    assert result['generated_token_ids'] == expected_ids == control['generated_token_ids']
    for key in ('prompt_token_ids', 'sampled_tokens_including_stop', 'finish_reasons', 'stop_token_ids'):
        assert result[key] == control[key]
    assert len(evidence['trace']) == limit and 'head_isolation' in evidence
    evidence['instrumented'] = result
    event('instrumented_request_complete')
finally:
    model.close()
    event('closed')
evidence['diagnostic_complete'] = True
save()
print('BATCH_DRIFT_DIAGNOSTIC_COMPLETE', batch, flush=True)
