"""Reject corrupt trained scheduler evidence without modifying originals."""
import copy
import json
from pathlib import Path
import shutil
import sys
import tempfile
from audit_scheduled_mtp_coding import audit

original, coding, http, output = map(Path, sys.argv[1:5])
rejected = []
with tempfile.TemporaryDirectory(prefix='mtp-scheduled-audit-') as temporary:
    root = Path(temporary)/original.name
    (root/'engine').mkdir(parents=True)
    for name in ('provenance.json', 'tasks.json', 'evaluation.json', 'engine/provenance.json',
                 'engine/result.json', 'engine/result.json.inventory.json'):
        shutil.copy2(original/name, root/name)
    shutil.copytree(original/'responses', root/'responses')
    log = root.parent/(root.name+'.log')
    shutil.copy2(original.parent/log.name, log)
    source_log = log.read_text()
    result_path = root/'engine/result.json'
    baseline = json.loads(result_path.read_text())
    assert audit(root, coding, http)['status'] == 'passed'
    changes = [
        ('missing task', lambda d: d['tasks'].pop()),
        ('changed prompt', lambda d: d['tasks'][0]['prompt'].__setitem__(0, 0)),
        ('changed output', lambda d: d['tasks'][0]['generated_ids'].__setitem__(1, 0)),
        ('changed native emission', lambda d: d['frames'][0]['requests'][0]['emitted'].__setitem__(0, 0)),
        ('stale lease', lambda d: d['frames'][0]['requests'][0].__setitem__('epoch', 0)),
        ('duplicate decode row', lambda d: next(f for f in d['frames'] if len(f['requests']) == 2)['requests'][1].__setitem__('row', next(f for f in d['frames'] if len(f['requests']) == 2)['requests'][0]['row'])),
        ('wrong budget', lambda d: d['frames'][0]['requests'][0].__setitem__('remaining_before', 1)),
        ('changed accepted count', lambda d: d['frames'][0]['generation'].__setitem__('accepted_drafts', 99)),
        ('unexpected recursive draft', lambda d: d['frames'][0]['generation'].__setitem__('recursive_draft_calls', 1)),
        ('auxiliary copy fallback', lambda d: d['frames'][0]['auxiliary'].__setitem__('kv_inplace_updates', 0)),
        ('missing hidden pooling', lambda d: d['frames'][0].__setitem__('hidden_pool_bindings', 0)),
        ('missing decode finish', lambda d: d['frames'][-1]['requests'][0].__setitem__('finished', False)),
        ('noncontiguous prefill', lambda d: d['prefills'][0].__setitem__('offset', 1)),
        ('false prefill finish', lambda d: d['prefills'][0].__setitem__('final', not d['prefills'][0]['final'])),
        ('missing combined frames', lambda d: d.__setitem__('combined_frames', 0)),
        ('missing prediction refresh', lambda d: d.__setitem__('prediction_refreshes', 0)),
        ('duplicate raw weights', lambda d: d['prepared_memory'].__setitem__('garnet_live_bf16_weight_owners', 16)),
        ('over-budget memory', lambda d: d['prepared_memory'].__setitem__('metal_allocated_bytes', 2**40)),
        ('missing deferred packs', lambda d: d.__setitem__('deferred_auxiliary_pack_slots', [0, 0])),
    ]
    if 'linear_chunk_state_inplace' in baseline.get('native_backend', ''):
        changes.append(('missing private recurrent updates', lambda d: d['frames'][0].__setitem__('linear_state_inplace_updates', 0)))
    for label, change in changes:
        changed = copy.deepcopy(baseline)
        change(changed)
        result_path.write_text(json.dumps(changed))
        try:
            audit(root, coding, http)
        except ValueError:
            rejected.append(label)
        else:
            raise AssertionError('Corrupted evidence admitted: '+label)
    result_path.write_text(json.dumps(baseline))
    inventory_path = root/'engine/result.json.inventory.json'
    inventory = json.loads(inventory_path.read_text())
    changed = copy.deepcopy(inventory)
    next(r for r in changed['references'] if r['tag'] == 'auxiliary_1' and r['role'] == 'weight_record')['buffer_id'] = 'unshared'
    inventory_path.write_text(json.dumps(changed))
    try:
        audit(root, coding, http)
    except ValueError:
        rejected.append('unshared auxiliary buffer')
    else:
        raise AssertionError('Unshared auxiliary weights admitted')
    inventory_path.write_text(json.dumps(inventory))
    lines = source_log.splitlines()
    for i, line in enumerate(lines):
        if line.startswith('scheduled_mtp_closed '):
            closed = json.loads(line.split(' ', 1)[1]);closed['garnet_live_matrix_owners'] = 1
            lines[i] = 'scheduled_mtp_closed '+json.dumps(closed)
    log.write_text('\n'.join(lines)+'\n')
    try:
        audit(root, coding, http)
    except ValueError:
        rejected.append('leaked native owner')
    else:
        raise AssertionError('Native owner leak admitted')
    log.write_text(source_log)
    assert audit(root, coding, http)['status'] == 'passed'
report = dict(status='passed', rejected=rejected, count=len(rejected), scope='CPU corruption tests; no inference or performance measurement')
with output.open('x') as stream:
    stream.write(json.dumps(report, indent=2)+'\n')
print(json.dumps(report))
