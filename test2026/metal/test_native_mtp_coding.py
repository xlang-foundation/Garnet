"""Rehash pinned assets and replay all eight frozen coding tasks with native MTP.

Arguments: runtime, build, source, asset manifest, auxiliary checkpoint,
quality manifest, reference worker JSONL, NEW output directory.
No comparator rerun and no model performance claim.
"""
import hashlib
import json
from pathlib import Path
import subprocess
import sys

runtime, build, source, assets, auxiliary, quality, reference, destination = sys.argv[1:9]
out = Path(destination)
out.mkdir(exist_ok=False)
manifest = json.loads(Path(quality).read_text())
assert len(manifest['plans']) == 8
assert manifest['suite_sha256'] == '675ea667c357f361ce39e3863b61193dc6b32f4a229eb7d29a17803b75408786'
assert hashlib.sha256(Path(auxiliary).read_bytes()).hexdigest() == 'dec6630788d1ba4557230ca02b5df2db239f2b2bbcfff1c6742776e231e7540b'
references = {}
for line in Path(reference).read_text().splitlines():
    event = json.loads(line)
    try:
        event = json.loads(event['line'])
    except (ValueError, KeyError):
        continue
    if isinstance(event, dict) and event.get('event') == 'completed':
        key = tuple(event['prompt_ids'])
        assert key not in references
        references[key] = event
assert len(references) == 8
plans = []
files = [Path(__file__), Path(__file__).with_name('run_native_mtp_coding.py'), Path(quality), Path(reference), Path(auxiliary)]
for plan in manifest['plans']:
    workload, cases = Path(plan['workload']), Path(plan['cases'])
    task = json.loads(workload.read_text())
    assert hashlib.sha256(cases.read_bytes()).hexdigest() == task['cases_sha256'] == plan['cases_sha256']
    assert task['suite_sha256'] == manifest['suite_sha256']
    assert len(task['requests']) == 1 and task['requests'][0]['prompt_tokens'] == plan['prompt_tokens']
    ids = task['requests'][0]['token_ids']
    expected = references[tuple(ids)]
    assert expected['finish_reason'] == 'stop' and expected['cached_tokens'] == 0
    plans.append(dict(id=plan['id'], prompt_ids=ids, max_output=task['max_output_tokens'],
                      reference_ids=expected['generated_ids'], reference_text=expected['text']))
    files.extend([workload, cases])
fixture = out/'coding.json'
fixture.write_text(json.dumps(dict(plans=plans, responses=str(out/'responses')))+'\n')
command = [sys.executable, str(Path(__file__).with_name('test_trained_verification_forward.py')),
           runtime, build, source, assets, '4096', str(out/'engine'), 'run_native_mtp_coding.py', auxiliary, str(fixture)]
files += [Path(build)/'libgarnet.dylib', fixture, Path(source)/'xModel/qwen38/vl_27b/metal_mtp_study.py',
          Path(source)/'xModel/qwen38/vl_27b/metal_mtp.py']
(out/'provenance.json').write_text(json.dumps(dict(argv=command, sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}, benchmark=False), indent=2)+'\n')
print('native_mtp_coding_frozen', len(plans), 'tasks', flush=True)
subprocess.run(command, check=True)
