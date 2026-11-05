"""Reject corrupted coding/transaction/weight evidence without changing originals."""
import copy
import json
from pathlib import Path
import shutil
import sys
import tempfile
from audit_native_mtp_coding import audit, digest

original = Path(sys.argv[1]).resolve()
with tempfile.TemporaryDirectory(prefix='mtp-coding-audit-') as temporary:
    root = Path(temporary)/original.name
    (root/'engine').mkdir(parents=True)
    for name in ['provenance.json', 'coding.json', 'evaluation.json', 'engine/provenance.json', 'engine/result.json']:
        shutil.copy2(original/name, root/name)
    shutil.copytree(original/'responses', root/'responses')
    log = root.parent/(root.name+'.log')
    shutil.copy2(original.parent/log.name, log)
    source_log = log.read_text()
    result_file = root/'engine/result.json'
    baseline = json.loads(result_file.read_text())
    inventory = json.loads(Path(baseline['inventory']).read_text())
    baseline['inventory'] = str(root/'inventory.json')
    result_file.write_text(json.dumps(baseline))
    (root/'inventory.json').write_text(json.dumps(inventory))
    assert audit(root)['status'] == 'passed'
    changes = [
        ('missing task', lambda r: r['tasks'].pop(), 'Complete eight tasks'),
        ('changed prompt', lambda r: r['tasks'][0]['prompt_ids'].__setitem__(0, 0), 'Frozen task identity'),
        ('changed complete output', lambda r: r['tasks'][0]['generated_ids'].__setitem__(0, 0), 'Complete frozen output IDs'),
        ('changed emitted token', lambda r: r['tasks'][0]['transactions'][0]['emitted'].__setitem__(0, 0), 'Every native emitted ID reconstructed'),
        ('wrong output budget', lambda r: r['tasks'][0]['transactions'][0].__setitem__('remaining_before', 1), 'Output budget'),
        ('wrong accepted count', lambda r: r['tasks'][0]['transactions'][0]['stats'].__setitem__('accepted_drafts', 99), 'Draft acceptance'),
        ('padded draft profile', lambda r: r['tasks'][0]['transactions'][0]['stats'].__setitem__('draft_profile_chunk', 4), 'Prepared native draft'),
        ('missing EOS', lambda r: r['tasks'][0]['transactions'][-1].__setitem__('eos', False), 'EOS/final'),
        ('padded prefill history', lambda r: r['tasks'][0]['prefill'].__setitem__('committed', [128]), 'Actual ragged prefill'),
        ('duplicate auxiliary weights', lambda r: r['prepared_memory'].__setitem__('garnet_live_bf16_weight_owners', 24), 'No duplicated raw'),
        ('over-budget prepared allocation', lambda r: r['prepared_memory'].__setitem__('metal_allocated_bytes', 2**40), 'Prepared allocation'),
    ]
    rejected = []
    for label, mutate, wanted in changes:
        changed = copy.deepcopy(baseline)
        mutate(changed)
        result_file.write_text(json.dumps(changed))
        try:
            audit(root)
        except ValueError as error:
            assert wanted in str(error), (label, str(error))
            rejected.append(label)
        else:
            raise AssertionError('Corrupted evidence admitted: '+label)
    result_file.write_text(json.dumps(baseline))
    changed = copy.deepcopy(inventory)
    ref = next(r for r in changed['references'] if r['tag'] == 'auxiliary_1' and r['role'] == 'weight_record')
    ref['buffer_id'] = 'changed-buffer'
    (root/'inventory.json').write_text(json.dumps(changed))
    try:
        audit(root)
    except ValueError as error:
        assert 'All fifteen auxiliary weight buffers shared' in str(error), str(error)
        rejected.append('unshared auxiliary weight')
    else:
        raise AssertionError('Unshared auxiliary weight admitted')
    (root/'inventory.json').write_text(json.dumps(inventory))
    lines = source_log.splitlines()
    for i, line in enumerate(lines):
        if line.startswith('native_mtp_coding_closed '):
            memory = json.loads(line.split(' ', 1)[1])
            memory['garnet_live_bf16_weight_owners'] = 1
            lines[i] = 'native_mtp_coding_closed '+json.dumps(memory)
    log.write_text('\n'.join(lines)+'\n')
    try:
        audit(root)
    except ValueError as error:
        assert 'Closed native owner' in str(error), str(error)
        rejected.append('leaked auxiliary owner')
    else:
        raise AssertionError('Leaked auxiliary owner admitted')
    log.write_text(source_log)
    assert audit(root)['status'] == 'passed'
report = dict(status='passed', corruption_checks=len(rejected), rejected=rejected,
              test_sha256=digest(__file__), auditor_sha256=digest(Path(__file__).with_name('audit_native_mtp_coding.py')))
with Path(sys.argv[2]).open('x') as stream:
    json.dump(report, stream, indent=2)
    stream.write('\n')
print(json.dumps(report))
