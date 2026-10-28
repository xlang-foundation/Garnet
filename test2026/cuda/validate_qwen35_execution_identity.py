"""Check frozen execution sources between model processes, without scanning weights.

Full checkpoint digests are captured separately before dispatch. This guard only
checks their saved size/mtime and does not claim protection against deliberate
same-metadata mutation. It must not be run during a resident model or timing role.
"""
import argparse
import hashlib
import json
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument('identity', type=Path)
a = p.parse_args()
identity = json.loads(a.identity.read_text(encoding='utf-8'))
root = a.identity.resolve().parent
repo = Path(identity.get('source_root', 'D:/CantorAI/Garnet'))


def check(path, expected):
    with path.open('rb') as stream:
        actual = hashlib.file_digest(stream, 'sha256').hexdigest()
    assert actual.lower() == expected.lower(), f'Execution file changed: {path}'


for name, expected in identity['source_sha256'].items():
    check(repo / name, expected)
for name, expected in identity['orchestration_sha256'].items():
    check(root / name, expected)
for name, expected in identity.get('external_input_sha256', {}).items():
    check(Path(name), expected)
if 'installed_vllm_cache_source' in identity:
    installed = identity['installed_vllm_cache_source']
    for name, expected in installed['sha256'].items():
        check(Path(installed['root']) / name, expected)
for name, key in (
    ('build-vs18/bin/garnet.dll', 'garnet_binary_sha256'),
    ('build-vs18/xlang3/xlang3.exe', 'xlang_binary_sha256'),
    ('build-vs18/xlang3/xlang3_runtime.dll', 'xlang_runtime_dll_sha256'),
):
    check(root / name, identity[key])
check(root / 'inputs/matched512-batches/manifest.json',
      identity['matched512_dataset_manifest_sha256'])
assert identity.get('checkpoint_stat_guard'), 'Missing pre-dispatch checkpoint metadata'
for name, expected in identity['checkpoint_stat_guard'].items():
    stat = (root / name).stat()
    assert (stat.st_size, stat.st_mtime_ns) == tuple(expected), f'Checkpoint metadata changed: {name}'
print('QWEN35_FROZEN_EXECUTION_IDENTITY_PASS')
