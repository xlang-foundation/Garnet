"""Freeze installed program identities and fixed real-image smoke requests."""
import argparse
import hashlib
import json
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('root', type=Path)
parser.add_argument('package', type=Path)
args = parser.parse_args()
root = args.root.resolve()
package = args.package.resolve()
repo = Path(__file__).resolve().parents[2]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


programs = {}
for source in sorted((repo / 'xModel').rglob('*')):
    if not source.is_file() or source.suffix not in ('.py', '.json') or source.name.startswith('debug_'):
        continue
    relative = source.relative_to(repo / 'xModel')
    for base in ('xModel', 'models'):
        target = package / base / relative
        assert target.is_file() and sha(target) == sha(source), ('Installed asset differs', target)
        programs[str(target.relative_to(package))] = sha(target)
adapter = package / 'runtime/cuda_serving.py'
assert sha(adapter) == sha(repo / 'runtime/cuda_serving.py')
programs['runtime/cuda_serving.py'] = sha(adapter)
worker = root / 'test_qwen35_packaged_serving.py'
source = Path(__file__).with_name(worker.name)
assert not worker.exists(), 'Preserve packaged worker'
worker.write_bytes(source.read_bytes())
requests = []
for size, batch in (('2b', 1), ('4b', 8)):
    reference = root / f'matched512-garnet-{size}-b{batch}-study-activation-arena-b1.json'
    rows = json.loads(reference.read_text())
    assert len(rows) == 28
    row = next(value for value in rows if value['trial'] == 2 and value['workload_group'] == 0)
    assert row['batch'] == batch and all(len(ids) >= 16 for ids in row['generated_token_ids'])
    paths = [root / 'inputs/matched512-batches/2/0' / str(index) / name
             for index, name in enumerate(row['images'])]
    assert [sha(path) for path in paths] == row['input_sha256']
    name = 'Qwen3.5-' + size.upper()
    request = {'package_root': str(package), 'forbidden_source_root': str(repo),
               'model_root': str(root / 'models' / name), 'model_id': name,
               'profile': row['profile'], 'prompts': ['Describe the main objects in this image in one sentence.'] * batch,
               'images': [str(path) for path in paths], 'input_sha256': row['input_sha256'],
               'output_limit': 16, 'expected_token_prefix': [ids[:16] for ids in row['generated_token_ids']],
               'reference': {'file': reference.name, 'sha256': sha(reference), 'trial': 2, 'workload_group': 0}}
    output = root / f'packaged-serving-{size}-b{batch}-request.json'
    assert not output.exists(), 'Preserve packaged request'
    output.write_text(json.dumps(request, indent=2), encoding='utf-8')
    requests.append({'file': output.name, 'sha256': sha(output)})
record = root / 'qwen35-installed-program-identity.json'
assert not record.exists(), 'Preserve installed asset identity'
record.write_text(json.dumps({'package_root': str(package), 'program_sha256': programs,
                            'worker_sha256': sha(worker), 'request_files': requests,
                            'scope': 'Installed program bytes and fixed smoke inputs only. No model execution, quality or clean-machine deployment claim.'}, indent=2), encoding='utf-8')
print(record)
