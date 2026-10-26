"""Real, non-replicated GDN Z fixture and independent CPU word replay.

This is a diagnostic matrix control. It admits no full-model or speed claim.
"""
import hashlib
import json
import struct

import numpy as np

from .gdn_chunk_stage_replay import array
from .prefill128_fp8_projection import input_oracle

N, K, M = 6144, 5120, 128
sha = lambda raw: hashlib.sha256(raw).hexdigest()


def bf16_output(raw):
    words = np.frombuffer(raw, dtype='<u4')
    assert len(raw) == M * N * 4 and np.isfinite(words.view('<f4')).all()
    assert np.all((words & 65535) == 0), 'Projection must be BF16 restored to F32'
    return (words >> 16).astype('<u2').tobytes()


def vectors(report, read, layer):
    """Caller first independently verifies the complete parent archive/proof."""
    assert type(layer) is int and layer in (0, 1)
    selected = {row['tokens']: row for row in report['cases']
                if row['layer'] == layer and row['diagnostic'] is True}
    assert set(selected) == {32, 128}
    assert all(row['primary_matches_ordinary'] is True for row in selected.values())
    small, large = selected[32]['repeats'][0], selected[128]['repeats'][0]
    assert len(small) == 4 and len(large) == 1
    assert [row['start'] for row in small] == [0, 32, 64, 96]
    assert large[0]['start'] == 0
    x32 = b''.join(array(row['arrays']['input_norm'], read) for row in small)
    x128 = array(large[0]['arrays']['input_norm'], read)
    assert x32 == x128 and len(x128) == M * K * 4
    assert x128 != x128[:16 * K * 4] * 8, 'Real128 input may not repeat16 rows'
    z32 = b''.join(array(row['arrays']['z'], read) for row in small)
    z128 = array(large[0]['arrays']['z'], read)
    result = {'input-m128-k5120.f32': x128,
              'expected-four-m32-n6144.bf16': bf16_output(z32),
              'expected-m128-n6144.bf16': bf16_output(z128)}
    different = np.flatnonzero(np.frombuffer(result['expected-four-m32-n6144.bf16'], dtype='<u2') !=
                              np.frombuffer(result['expected-m128-n6144.bf16'], dtype='<u2'))
    assert different.size > 0
    return result, {'layer': layer, 'input_lineage': selected[128]['input_lineage'],
                    'input_words': M * K, 'output_words_per_profile': M * N,
                    'different_bf16_words': int(different.size),
                    'first_different_word': int(different[0])}


def checkpoint_tensor(checkpoint, index, key, dtype, shape):
    """Read the actual indexed tensor only, checking all dimensions and bounds."""
    shard_name = index['weight_map'][key]
    assert isinstance(shard_name, str) and '/' not in shard_name and '\\' not in shard_name
    shard = checkpoint / shard_name
    with shard.open('rb') as stream:
        raw_length = stream.read(8)
        assert len(raw_length) == 8
        length = struct.unpack('<Q', raw_length)[0]
        assert 0 < length < (64 << 20) and 8 + length <= shard.stat().st_size
        header = stream.read(length)
        item = json.loads(header)[key]
        assert item['dtype'] == dtype and item['shape'] == shape
        begin, end = item['data_offsets']
        assert type(begin) is type(end) is int and 0 <= begin < end
        assert 8 + length + end <= shard.stat().st_size
        stream.seek(8 + length + begin)
        raw = stream.read(end - begin)
    assert len(raw) == end - begin
    if dtype == 'F8_E4M3':
        assert len(raw) == N * K and not np.any((np.frombuffer(raw, dtype='u1') & 127) == 127)
    else:
        assert dtype == 'F32' and shape == [] and len(raw) == 4
        value = struct.unpack('<f', raw)[0]
        assert np.isfinite(value) and value > 0
    return raw, {'key': key, 'shard': shard_name, 'header_sha256': sha(header), **item}


def replay(read_fixture, read_actual, mode):
    """Require both archived profiles before an M32-shared intervention."""
    assert mode in ('control', 'shared')
    source = read_fixture('input-m128-k5120.f32')
    expected32 = read_fixture('expected-four-m32-n6144.bf16')
    expected128 = read_fixture('expected-m128-n6144.bf16')
    assert len(source) == M * K * 4 and len(expected32) == len(expected128) == M * N * 2
    assert source != source[:16 * K * 4] * 8
    scale_raw = read_fixture('input-scale.f32')
    assert len(scale_raw) == 4
    cast, packed = input_oracle(source, struct.unpack('<f', scale_raw)[0])
    assert read_actual('cast-m128.bf16') == cast
    assert read_actual('quantized-m128.e4m3') == packed
    rows = []
    for profile in ('four-m32', 'm128'):
        expected = expected32 if profile == 'four-m32' or mode == 'shared' else expected128
        previous = None
        for repeat in (0, 1):
            actual = read_actual(profile + '-r' + str(repeat) + '.bf16')
            assert len(actual) == M * N * 2
            words = np.frombuffer(actual, dtype='<u2')
            assert np.all((words & 0x7f80) != 0x7f80)
            assert actual == expected, (mode, profile, repeat, 'Complete output mismatch')
            if previous is not None:
                assert actual == previous
            previous = actual
            rows.append({'profile': profile, 'repeat': repeat, 'words': M * N,
                         'complete_expected_output_exact': True})
    return {'status': 'real_gdn_z_projection_' + mode + '_words_replayed',
            'cases': rows, 'complete_bf16_output_words_replayed': 4 * M * N,
            'independent_input_cast_and_quantization_exact': True,
            'real128_rows_not_repeated16': True,
            'full_model_cause_established': False,
            'model_or_quality_or_performance_qualified': False}
