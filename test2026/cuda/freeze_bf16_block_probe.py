"""Freeze the standalone source/fixtures before compiling; no runtime changes."""
import argparse
import datetime
import hashlib
import json
import zipfile
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument('root', type=Path)
p.add_argument('--validate', type=Path)
p.add_argument('--attempt', default='v1')
a = p.parse_args()


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


if a.validate:
    manifest = json.loads(a.validate.read_text(encoding='utf-8-sig'))
    for filename, expected in manifest['files_sha256'].items():
        assert digest(Path(filename)) == expected, 'Frozen component input changed: ' + filename
    assert digest(a.root / 'build-vs18/bin/garnet.dll') == manifest['unchanged_garnet_dll_sha256']
    print('BF16_BLOCK_FROZEN_INPUTS_PASS')
    raise SystemExit(0)
manifest_path = a.root / f'lossless-bf16-projection-{a.attempt}-before-build-identity.json'
archive_path = a.root / f'lossless-bf16-projection-{a.attempt}-before-build-source-fixtures.zip'
assert not manifest_path.exists() and not archive_path.exists(), 'Preserve component source identity'
repo = a.root / 'Garnet'
members = [repo / 'test2026/cuda' / name for name in (
    'bf16_block_codec.py', 'test_bf16_block_codec.py', 'prepare_bf16_block_projection.py',
    'probe_bf16_block_projection.cu', 'freeze_bf16_block_probe.py',
    'summarize_bf16_block_projection.py')]
members += [repo / 'src/cuda/bf16_projection.cu']
members += [a.root / name for name in ('build-bf16-block-probe.cmd', 'run-bf16-block-probe.ps1','wait-qwen35-original-idle.ps1',
    'lossless-bf16-codec-v1-tests.json', 'lossless-bf16-projection-v1-fixtures.json',
    'lossless-bf16-projection-v1-prospective-design.json', 'compact-kv-completed-study-v1-identity.json')]
if a.attempt != 'v1':
    members += [a.root / name for name in ('lossless-bf16-projection-v1-controller-exit.json',
        'lossless-bf16-projection-v1-before-build-identity.json','lossless-bf16-projection-v1-plain-exit.json',
        'lossless-bf16-projection-v1-plain.log','lossless-bf16-projection-v1-plain.jsonl','bf16-block-projection-probe.exe')]
fixtures = json.loads((a.root / 'lossless-bf16-projection-v1-fixtures.json').read_text(encoding='utf-8'))
for record in fixtures['fixtures']:
    assert record['all_words_byte_roundtrip_exact']
    for kind in ('raw', 'packed'):
        path = Path(record[kind + '_file'])
        assert digest(path) == record[kind + '_sha256']
        members.append(path)
manifest = {'captured_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
    'files_sha256': {str(path): digest(path) for path in members},
    'unchanged_garnet_dll_sha256': digest(a.root / 'build-vs18/bin/garnet.dll'),
    'gpu_uuid': 'GPU-472bb21a-0ca8-9d60-47a0-c527bb3d2d81',
    'scope': 'Exact standalone probe sources and canonical prevalidated learned/bit-pattern '
        'fixtures. Production GEMM source reused; Garnet DLL is neither linked nor rebuilt. '
        'CPU byte tests passed before native dispatch. No model inference or memory claim.'}
manifest_path.write_text(json.dumps(manifest, indent=2), encoding='utf-8')
with zipfile.ZipFile(archive_path, 'x', compression=zipfile.ZIP_STORED) as archive:
    archive.writestr('identity.json', json.dumps(manifest, indent=2))
    for path in members:
        archive.write(path, str(path.relative_to(a.root)).replace('\\', '/'))
print(manifest_path)
