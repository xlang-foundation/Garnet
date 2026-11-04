"""Corrupt copied HTTP/native evidence; never alter originals or start inference."""
import json
from pathlib import Path
import shutil
import sys
import tempfile
from audit_local_coder_mtp_http import audit, digest

original, qualified, reference = map(Path, sys.argv[1:4])
rejected = []
with tempfile.TemporaryDirectory(prefix='mtp-http-audit-') as temporary:
    root = Path(temporary)/'admission'
    shutil.copytree(original, root)
    assert audit(root, qualified, reference)['status'] == 'passed'
    log = root/'server/worker-stdout.jsonl'
    original_log = log.read_text()

    def native_change(kind, mutate):
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
                return
        raise AssertionError('Missing real event '+kind)

    native_changes = [
        ('native emitted ID', 'mtp_transaction', lambda e:e['emitted'].__setitem__(0, 0), 'Stream exactly native emitted IDs'),
        ('native consumed count', 'mtp_transaction', lambda e:e['stats'].__setitem__('consumed', [99]), 'Native target consumption'),
        ('padded draft profile', 'mtp_transaction', lambda e:e['stats'].__setitem__('draft_profile_chunk', 4), 'Prepared T1/T4 native selection'),
        ('padded prefill length', 'mtp_prefill', lambda e:e.__setitem__('actual_input_tokens', 128), 'Actual prefill length'),
        ('wrong native prompt', 'completed', lambda e:e['prompt_ids'].__setitem__(0, 0), 'Full frozen HTTP/native prompt IDs'),
        ('native usage count', 'token', lambda e:e.__setitem__('sampled_tokens', 2), 'Native token/usage sequence'),
        ('leaked auxiliary owner', 'closed', lambda e:e['memory'].__setitem__('garnet_live_bf16_weight_owners', 1), 'Closed owner'),
        ('false native batching', 'ready', lambda e:e.__setitem__('native_continuous_batching', True), 'Honest B1 uncached text profile'),
    ]

    def rejects(label, wanted):
        try:
            audit(root, qualified, reference)
        except ValueError as error:
            assert wanted in str(error), (label, str(error))
            rejected.append(label)
        else:
            raise AssertionError('Corrupted evidence admitted: '+label)

    for label, kind, mutate, wanted in native_changes:
        native_change(kind, mutate)
        rejects(label, wanted)
        log.write_text(original_log)
    stream = root/'responses/merge-closed-intervals/request-0.events.jsonl'
    original_stream = stream.read_text()
    data = [json.loads(line) for line in original_stream.splitlines()]
    event = json.loads(data[0]['data'])
    event['token_id'] = 0
    data[0]['data'] = json.dumps(event)
    stream.write_text('\n'.join(json.dumps(e) for e in data)+'\n')
    rejects('HTTP token ID', 'Full frozen HTTP/native output IDs')
    stream.write_text(original_stream)
    stream.write_text(original_stream+json.dumps(dict(monotonic=0, data='[DONE]'))+'\n')
    rejects('duplicate HTTP DONE', 'One terminal HTTP DONE')
    stream.write_text(original_stream)
    data = [json.loads(line) for line in original_stream.splitlines()]
    event = json.loads(data[0]['data'])
    event['usage']['completion_tokens'] = 99
    data[0]['data'] = json.dumps(event)
    stream.write_text('\n'.join(json.dumps(e) for e in data)+'\n')
    rejects('HTTP token usage', 'Monotone per-token HTTP usage')
    stream.write_text(original_stream)
    worker = root/'sources/local_coder_mtp_worker.py'
    original_worker = worker.read_bytes()
    worker.write_bytes(original_worker+b'\n')
    rejects('changed worker source', 'Frozen implementation/profile/workload changed')
    worker.write_bytes(original_worker)
    wire = root/'responses/merge-closed-intervals/request-0.request.json'
    original_wire = wire.read_bytes()
    wire.write_bytes(b'{}')
    rejects('actual submitted request bytes', 'Actual submitted request bytes')
    wire.write_bytes(original_wire)
    exit_file = root/'server/worker-exit.json'
    original_exit = exit_file.read_text()
    exit_file.write_text(json.dumps(dict(returncode=-9, expected_close=True)))
    rejects('unclean worker shutdown', 'Native worker clean exit')
    exit_file.write_text(original_exit)
    assert audit(root, qualified, reference)['status'] == 'passed'
report = dict(status='passed', corruption_checks=len(rejected), rejected=rejected,
              source_sha256=digest(__file__), auditor_sha256=digest(Path(__file__).with_name('audit_local_coder_mtp_http.py')))
with Path(sys.argv[4]).open('x') as stream:
    json.dump(report, stream, indent=2)
    stream.write('\n')
print(json.dumps(report))
