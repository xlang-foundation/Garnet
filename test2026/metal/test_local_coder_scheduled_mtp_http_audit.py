"""Corrupt copied B2 HTTP evidence; never modify originals or start inference."""
import json
from pathlib import Path
import shutil
import sys
import tempfile
from audit_local_coder_scheduled_mtp_http import audit

original, qualified, reference, report = map(Path, sys.argv[1:5])
rejected = []
with tempfile.TemporaryDirectory(prefix='scheduled-mtp-http-audit-') as temporary:
    root = Path(temporary)/'admission'
    shutil.copytree(original, root)
    assert audit(root, qualified, reference)['status'] == 'passed'
    log = root/'server/worker-stdout.jsonl'
    original_log = log.read_text()

    def rejects(label):
        try:
            audit(root, qualified, reference)
        except ValueError:
            rejected.append(label)
        else:
            raise AssertionError('Corrupted serving evidence admitted: '+label)

    changes = [
        ('native emission', 'mtp_transaction', lambda e: e['transaction']['emitted'].__setitem__(0, 0)),
        ('native consumed count', 'mtp_batch_transaction', lambda e: e['result']['requests'][0].__setitem__('consumed', 99)),
        ('native budget', 'mtp_batch_transaction', lambda e: e['result']['requests'][0].__setitem__('remaining_before', 1)),
        ('stale native lease', 'mtp_batch_transaction', lambda e: e['result']['requests'][0].__setitem__('epoch', 0)),
        ('invalid physical row', 'mtp_schedule', lambda e: e['plan']['batches'][0]['requests'][0].__setitem__('row', 99)),
        ('wrong active mask', 'mtp_schedule', lambda e: e['plan']['batches'][0]['active_rows'].__setitem__(0, 1-e['plan']['batches'][0]['active_rows'][0])),
        ('missing hidden pooling', 'mtp_batch_transaction', lambda e: e.__setitem__('hidden_pool_bindings', 0)),
        ('missing recurrent in-place updates', 'mtp_batch_transaction', lambda e: e.__setitem__('linear_state_inplace_updates', 0)),
        ('auxiliary copy fallback', 'mtp_batch_transaction', lambda e: e['auxiliary'].__setitem__('kv_inplace_updates', 0)),
        ('premature native completion', 'mtp_batch_transaction', lambda e: e['result']['requests'][0].__setitem__('finished', True)),
        ('false batching metadata', 'ready', lambda e: e.__setitem__('native_continuous_batching', False)),
        ('wrong verification profile', 'ready', lambda e: e.__setitem__('verification_chunk', 4)),
        ('native token usage', 'token', lambda e: e.__setitem__('sampled_tokens', 99)),
        ('wrong full prompt', 'completed', lambda e: e['prompt_ids'].__setitem__(0, 0)),
        ('native owner leak', 'closed', lambda e: e['memory'].__setitem__('garnet_live_matrix_owners', 1)),
    ]
    for label, kind, mutate in changes:
        lines = original_log.splitlines()
        for i, line in enumerate(lines):
            outer = json.loads(line)
            try:
                event = json.loads(outer['line'])
            except ValueError:
                continue
            if isinstance(event, dict) and event.get('event') == kind:
                mutate(event)
                outer['line'] = json.dumps(event)
                lines[i] = json.dumps(outer)
                log.write_text('\n'.join(lines)+'\n')
                break
        else:
            raise AssertionError('Missing actual serving event: '+kind)
        rejects(label)
        log.write_text(original_log)
    stream = root/'responses/merge-closed-intervals/request-0.events.jsonl'
    original_stream = stream.read_text()
    for label, mutate in [
        ('HTTP token ID', lambda e: e.__setitem__('token_id', 0)),
        ('HTTP token usage', lambda e: e['usage'].__setitem__('completion_tokens', 99)),
    ]:
        data = [json.loads(line) for line in original_stream.splitlines()]
        i = next(i for i, row in enumerate(data) if row['data'] != '[DONE]' and 'token_id' in json.loads(row['data']))
        event = json.loads(data[i]['data']);mutate(event);data[i]['data'] = json.dumps(event)
        stream.write_text('\n'.join(json.dumps(e) for e in data)+'\n')
        rejects(label)
        stream.write_text(original_stream)
    stream.write_text(original_stream+json.dumps(dict(monotonic=0, data='[DONE]'))+'\n')
    rejects('duplicate HTTP DONE');stream.write_text(original_stream)
    for label, path, value in [
        ('changed frozen worker', root/'sources/local_coder_scheduled_mtp_worker.py', b'changed'),
        ('changed submitted wire bytes', root/'responses/merge-closed-intervals/request-0.request.json', b'{}'),
    ]:
        saved = path.read_bytes();path.write_bytes(value);rejects(label);path.write_bytes(saved)
    exit_path = root/'server/worker-exit.json';saved_exit = exit_path.read_text()
    changed = json.loads(saved_exit);changed['returncode'] = -2;exit_path.write_text(json.dumps(changed))
    rejects('abnormal native shutdown');exit_path.write_text(saved_exit)
    inventory_path = root/'memory-inventory.json';saved_inventory = inventory_path.read_text()
    changed = json.loads(saved_inventory)
    next(r for r in changed['references'] if r['tag'] == 'auxiliary_1' and r['role'] == 'weight_record')['buffer_id'] = 'unshared'
    inventory_path.write_text(json.dumps(changed));rejects('unshared auxiliary weights');inventory_path.write_text(saved_inventory)
    assert audit(root, qualified, reference)['status'] == 'passed'
with report.open('x') as stream:
    stream.write(json.dumps(dict(status='passed', count=len(rejected), rejected=rejected,
        scope='CPU-only corruption tests over actual collected serving evidence; no inference or performance measurement'), indent=2)+'\n')
print(json.dumps(dict(status='passed', count=len(rejected), rejected=rejected)))
