"""Step44/45 layer outputs, interpreted only after ordinary primary/cache parity."""
from .b4_k35_coordinate_trajectory import analyze as primary_analysis
from .decode_boundary_analysis import blob, difference

LAYERS = tuple(range(64))
STEPS = (44, 45)
ROW_BYTES = 4 * 5120 * 4


def layers(report, read, selected_key):
    batch = report['physical_batch']
    assert type(batch) is int and batch in (1, 4)
    assert len(report['rows']) == 1 and report['boundary_runs'] == 52
    profile = report['observer']
    assert set(profile) == {'layers', 'tokens', 'hidden', 'hidden_width', 'primary_width', 'wide_width', 'selected_steps'}
    assert profile['layers'] == list(LAYERS) and profile['selected_steps'] == list(STEPS)
    assert (profile['tokens'], profile['hidden'], profile['hidden_width']) == (4, 5120, 20480)
    assert type(profile['primary_width']) is int and profile['primary_width'] > 0
    assert profile['wide_width'] == profile['primary_width'] + 64 * 20480
    row = report['rows'][0]
    assert len(row['keys']) == batch and len(set(row['keys'])) == batch
    assert selected_key in row['keys']
    position = row['keys'].index(selected_key)
    assert (batch, position) in ((1, 0), (4, 2))
    entries = row['layer_observations']
    assert [e['step'] for e in entries] == list(STEPS)
    output = {}
    for entry in entries:
        step = entry['step']
        assert type(step) is int and entry['program_runs'] == step
        assert entry['consumed_tokens'] == [ids[step - 1] for ids in row['generated_ids']]
        assert set(entry['boundaries']) == {str(i) for i in LAYERS}
        assert len({v['file'] for v in entry['boundaries'].values()}) == 64
        arrays = {i: blob(entry['boundaries'][str(i)], read, batch * ROW_BYTES) for i in LAYERS}
        output[step] = {
            'consumed_token': entry['consumed_tokens'][position],
            'boundaries': {i: raw[position * ROW_BYTES:(position + 1) * ROW_BYTES] for i, raw in arrays.items()},
        }
    return output


def analyze(selection, key, reference, candidate, old_reference, old_candidate,
            read_reference, read_candidate, read_old_reference, read_old_candidate):
    # This must be the first numerical analysis. Added outputs can change TRT
    # tactics/fusions even when the arithmetic AST is identical.
    primary = primary_analysis(selection, key, reference, candidate, old_reference, old_candidate,
                               read_reference, read_candidate, read_old_reference, read_old_candidate)
    for report, parent in ((reference, old_reference), (candidate, old_candidate)):
        assert report['rows'][0]['key35_coordinate_trajectory'] == parent['rows'][0]['key35_coordinate_trajectory'], 'Observer changed ordinary coordinate timeline'
    a = layers(reference, read_reference, key)
    b = layers(candidate, read_candidate, key)
    rows = []
    for step in STEPS:
        assert a[step]['consumed_token'] == b[step]['consumed_token']
        changes = {str(i): difference(a[step]['boundaries'][i], b[step]['boundaries'][i], active_words=5120) for i in LAYERS}
        rows.append({
            'step': step, 'key': key, 'physical_row': 2,
            'consumed_token': a[step]['consumed_token'], 'all52_consumed_IDs_exact': True,
            'boundaries': changes,
            'first_differing_full_T4_layer': next((i for i in LAYERS if not changes[str(i)]['all_bytes_exact']), None),
            'first_differing_active_layer': next((i for i in LAYERS if changes[str(i)]['active_different_words']), None),
        })
    return {
        'status': 'step44_step45_all64_layer_outputs_compared_after_ordinary_primary_cache_timeline_parity',
        'ordinary_primary_cache_timeline': primary, 'rows': rows,
        'numerical_tolerance_applied': False, 'original_full4k_still_failed': True,
        'model_or_MTP_quality_HTTP_or_performance_qualified': False, 'two_times_target_achieved': False,
        'scope': 'Only selected row, ordinary decode44/45, complete T4 outputs after each layer. Active column0 established by frozen step/chunk source. Every original captured primary state/vector/53IDs/full8MiBcache and coordinate timeline must reproduce same-profile parents before interpretation. Observed layer boundary differences do not identify a unique operator, direct internal projection/normalization activation, or earliest global decoder drift. No old step<=7 protocol is relaxed.',
    }
