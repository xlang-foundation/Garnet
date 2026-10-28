"""B8 MTP admission retains the complete ordinary-model oracle, without timing."""
from pathlib import Path
import hashlib
import json
import zipfile
from .paged_batch8_model import same_state, same_case
from .verify_bf16_batch8_admission import replay_cases

PARENT = 'paged-batch8-model-preservation-attempt1'
BASELINE = '1a1d0229a1c0c24a8397a4dee7a614bcedf97b3bd3293e58d752ed7de30eb498'
sha = lambda raw: hashlib.sha256(raw).hexdigest()


def validate_parent(proof, workflow, library, archive_sha256):
    assert library == proof['library_sha256'] == workflow['library_sha256'] == BASELINE
    assert proof['status'] == 'paged_batch8_model_preservation_replayed'
    assert proof['archive_sha256'] == archive_sha256
    assert type(workflow['exit_code']) is int and workflow['exit_code'] == 0
    assert workflow['cleanup_finished'] is workflow['library_unchanged'] is True
    assert workflow['source_sha256'] == workflow['source_after_sha256']
    assert workflow['runtime_dependencies'] == workflow['runtime_dependencies_after']
    assert workflow['runtime_dependencies']['garnet.dll'] == library
    assert workflow['environment_before'] == workflow['environment_after']
    assert proof['preserved_model']['complete_states'] == 29 and proof['preserved_model']['raw_vectors'] == 58
    assert proof['preserved_model']['all_retained_states_vectors_tokens_exact'] is True
    assert proof['external_b1_b4_math_qualified'] is proof['mtp_qualified'] is proof['http_qualified'] is False
    assert proof['performance_admitted'] is proof['two_times_target_achieved'] is False


def require_parent(root, library):
    root = Path(root)
    path = root / 'results' / (PARENT + '-verified.json')
    proof = json.loads(path.read_bytes())
    archive = root / 'results' / (PARENT + '-evidence.zip')
    digest = sha(archive.read_bytes())
    with zipfile.ZipFile(archive) as z:
        manifest = json.loads(z.read('evidence-manifest.json'))
        assert z.testzip() is None and len(z.namelist()) == len(set(z.namelist())) == len(manifest['files']) + 1
        for name, expected in manifest['files'].items():
            assert sha(z.read(name)) == expected
        w = json.loads(z.read('results/' + PARENT + '/workflow.json'))
    assert w == json.loads((root / 'results' / PARENT / 'workflow.json').read_bytes())
    validate_parent(proof, w, library, digest)
    for name, expected in w['source_after_sha256'].items():
        assert sha((root / name).read_bytes()) == expected
    for name, expected in w['runtime_dependencies_after'].items():
        assert sha((root / 'build/bin' / name).read_bytes()) == expected
    for name, expected in w['environment_after'].items():
        assert sha(Path(name).read_bytes()) == expected
    return {'job': PARENT, 'archive_sha256': digest, 'proof_sha256': sha(path.read_bytes()),
            'ordinary_model_preserved': True, 'mtp_or_http_or_performance_admitted': False}


def replay_mtp(report, read, reference, read_reference):
    assert report['library_sha256'] == reference['library_sha256'] == BASELINE
    assert report['status'] == 'paged_batch8_mtp_collected_needs_replay'
    assert report['physical_batch'] == 8 and report['capacity'] == 4096 and report['mtp_executed'] is True
    assert all(report[key] is False for key in ('http_executed', 'quality_qualified', 'performance_admitted'))
    assert report['mtp_depth'] == 3 and report['draft_chunk'] == 4
    assert report['draft_builder_workspace_bytes'] == 128 << 20
    assert report['mtp_stop_ids'] == [] and report['prefix_replay'] is True
    assert len(report['cases']) == len(reference['cases']) == 9
    groups = []
    for i, (label, limit) in enumerate((('short-mixed', 8), ('full4k-text', 128), ('full4k-image', 128))):
        case = reference['cases'][i*3]
        assert case['group'] == label and case['repeat'] == 0 and case['order'] == list(range(8)) and case['actual_limit'] == limit
        groups.append((label, case['inputs'], limit))
    # Validate finite raw vectors, all128state components, every row/order and
    # actual prompt/output bounds in the unchanged independent ordinary oracle.
    replay_cases(reference, read_reference, groups)
    same_state(report['empty_state'], reference['empty_state'], read, read_reference)
    same_state(report['final_reset'], reference['final_reset'], read, read_reference)
    iterations = 0
    for case, ordinary in zip(report['cases'], reference['cases']):
        same_case(case, ordinary, read, read_reference)
        assert case['inverse_permutation_exact'] is True and case['performance_admitted'] is False
        result = case['mtp_result']
        limit = case['actual_limit']
        assert result['generated_token_ids'] == case['generated_ids']
        assert result['sampled_tokens_including_stop'] == [limit]*8
        assert result['finish_reasons'] == ['length']*8 and result['stop_token_ids'] == [None]*8
        assert result['decode_steps'] == limit-1
        metrics = result['speculative_metrics']
        totals = [0]*8
        assert metrics['iterations'] and metrics['target_verification_calls'] == len(metrics['iterations'])
        previous = [0]*8
        for step in metrics['iterations']:
            counts = step['processed_target_counts']
            sampled = step['sampled_so_far']
            assert len(counts) == len(sampled) == 8
            assert all(type(v) is int and 0 <= v <= 4 for v in counts)
            assert all(type(v) is int and previous[i] <= v <= limit for i, v in enumerate(sampled))
            totals = [a+b for a,b in zip(totals, counts)]
            previous = sampled
        assert totals == [limit-1]*8
        assert case['mtp_memory']['draft_builder_workspace_bytes'] == 128 << 20
        assert case['mtp_memory']['draft_chunk'] == 4 and case['mtp_memory']['target_prefill_chunk'] == 32
        iterations += len(metrics['iterations'])
    return {'status': 'paged_batch8_mtp_retained_ordinary_exact', 'cases': 9, 'complete_states': 29,
            'raw_vectors': 58, 'actual_mtp_iterations': iterations, 'all_target_states_vectors_tokens_exact': True,
            'external_b1_b4_math_qualified': False, 'broad_quality_qualified': False, 'http_qualified': False,
            'serving_memory_saving_qualified': False, 'performance_admitted': False, 'two_times_target_achieved': False}
