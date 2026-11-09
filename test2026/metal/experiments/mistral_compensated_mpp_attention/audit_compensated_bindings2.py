"""Check numerical evidence, exact model-fixture provenance, controls and hardware maps."""
from pathlib import Path
import sys
import hashlib
import json
import numpy as np

p = Path(sys.argv[1]).resolve() if len(sys.argv)>1 else Path(__file__).resolve().parent
load = lambda q: json.loads(q.read_text())
def sha(q):
    h = hashlib.sha256()
    with q.open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

f = p.parent / 'mistral-trained-attention-fixture1'
binding = load(f / 'replay-fixture1/fixture-bindings1.json')
assert binding['layers'] == [0, 15, 31]
assert sha(f / 'capture2/capture.json') == binding['source_capture_metadata_sha256']
assert sha(f / 'bindings2.json') == binding['model_bindings_sha256']
assert load(p / 'trained-status1.json')['returncode'] == 0
assert load(p / 'synthetic-status1.json')['returncode'] == 0
word_pairs = 0
fixtures = {}
for kind in ['inputs', 'outputs']:
    for name, record in binding[kind].items():
        original = f / 'replay-fixture1' / name
        assert sha(original) == record['sha256']
        if kind == 'inputs':
            assert sha(p / 'trained1' / name) == record['sha256']
        fixtures[name] = record['sha256']
for case in range(3):
    for variant in [0, 1]:
        name = f'case{case}-output{variant}.f32'
        assert sha(p / 'trained1' / name) == sha(f / 'replay1' / name), name
        word_pairs += (p / 'trained1' / name).stat().st_size // 4
reports = {}
for folder, trained in [('synthetic1', False), ('trained1', True)]:
    r = load(p / folder / 'correctness1.json')
    a = load(p / folder / 'audit1.json')
    assert r['actual_trained_operands'] is trained and not r['timing_admitted']
    assert r['compensated_half_layout_capacities'] == [8,16,16,8,16,16] * 32
    assert r['register_layout_status'] == [0] * 32
    assert r['prototype_input_domain_max_abs'] == 2048
    assert a['passed'] and all(a['qualified_variants'][i] for i in [0,1,3])
    assert a['original_bound'] == '1e-5 + 2e-4*abs(FP64 expected)'
    expected_rejected = {'0':[], '1':[], '2':[] if trained else [7], '3':[]}
    assert a['rejected_cases_by_variant'] == expected_rejected
    assert len(a['cases']) == (3 if trained else 10)
    assert a['fp64_values_per_variant'] == (18432 if trained else 33280)
    assert all(c['fp64_bound_fraction_max'][-1] <= 1 for c in a['cases'])
    for name, h in load(p / folder / 'payload-hashes1.json').items():
        assert sha(p / folder / name) == h, name
    reports[folder] = {'audit_sha256':sha(p / folder / 'audit1.json'),
                       'metadata_sha256':sha(p / folder / 'correctness1.json'),
                       'payload_manifest_sha256':sha(p / folder / 'payload-hashes1.json'),
                       'qualified_variants':a['qualified_variants']}
source = load(p / 'source-bindings2.json')
assert sha(p / 'compensated_attention1.mm') == source['host_sha256']
assert sha(p / 'combined2.metal') == source['shader_sha256']
timing = load(p / 'timing-host-bindings2.json')
assert sha(p / 'compensated_attention2.mm') == timing['timing_host_sha256']
assert sha(p / 'combined2.metal') == timing['candidate_shader_sha256']
result = dict(passed=True, exact_trained_control_word_pairs=word_pairs,
              trained_fixture_payloads_sha256=fixtures, correctness=reports,
              admitted_variants=[0,1,3], rejected_variant=2,
              source_bindings_sha256=sha(p / 'source-bindings2.json'),
              timing_bindings_sha256=sha(p / 'timing-host-bindings2.json'),
              original_bound_unchanged=True, native_integration=False,
              serving_speed_claim=False, goal_complete=False)
(p / 'bindings-audit2.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:result[k] for k in ['passed','exact_trained_control_word_pairs','admitted_variants']},indent=2))
