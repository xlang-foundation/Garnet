"""Read/copy the exact installed config/router sources; no model or GPU imports."""
from pathlib import Path
import argparse
import hashlib
import json

ROOT = Path('/mnt/d/CantorAI/5090-study-20261023')
METHODS = ROOT / 'Garnet/test2026/cuda/qwen38_5090'
SOURCE = Path('/opt/cantorai5090/vllm/lib/python3.12/site-packages/vllm')
sha = lambda raw: hashlib.sha256(raw).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    out = args.output.resolve()
    assert out.is_relative_to((ROOT / 'results').resolve()) and out.parent.is_dir() and not out.exists()
    pins_path = METHODS / 'evidence/vllm-fixed4k-config-schema-attempt1.json'
    pins = json.loads(pins_path.read_bytes())
    assert pins['status'] == 'installed_vllm_config_source_schema_read_cpu_only' and pins['source_modified'] is pins['gpu_executed'] is False
    assert pins['source_root'] == str(SOURCE) and len(pins['files']) == 5
    destination = out.parent / 'vllm-config-source'
    assert not destination.exists()
    rows = {}
    for name, row in pins['files'].items():
        source = SOURCE / name
        assert source.resolve().is_relative_to(SOURCE.resolve())
        raw = source.read_bytes()
        assert sha(raw) == row['sha256'], ('Installed config source changed', name)
        target = destination / name
        assert target.resolve().is_relative_to(destination.resolve())
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open('xb') as stream:
            stream.write(raw)
        rows[name] = {'sha256': sha(raw), 'archived_copy': target.relative_to(out.parent).as_posix()}
    result = {'status': 'installed_vllm_fixed4k_config_sources_verified', 'pins_sha256': sha(pins_path.read_bytes()),
              'source_modified': False, 'gpu_executed': False, 'files': rows}
    with out.open('xb') as stream:
        stream.write(json.dumps(result, indent=2).encode())
    print(json.dumps({'status': result['status'], 'source_files': len(rows)}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
