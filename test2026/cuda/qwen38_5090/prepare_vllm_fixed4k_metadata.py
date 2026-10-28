"""Check actual ready vLLM JSON config, version, capacity, source and owned PID."""
from pathlib import Path
import argparse
import hashlib
import json
import subprocess
from urllib.request import urlopen
from reference_fixed4k_vllm_identity import validate

sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--job', type=Path, required=True)
    parser.add_argument('--mode', choices=('ordinary',), required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    job = args.job.resolve()
    assert root == Path('D:/CantorAI/5090-study-20261023').resolve()
    assert job.is_relative_to((root / 'results').resolve()) and job.is_dir()
    target = job / 'metadata.json'
    assert not target.exists()
    methods = Path(__file__).resolve().parent
    launcher = methods / 'serve_vllm_fixed4k.sh'
    proof_path = root / 'results/fixed-length-http-study-attempt1-verified.json'
    proof = json.loads(proof_path.read_bytes())
    assert proof['status'] == 'fixed_length_http_study_replayed' and proof['native_fixed_4k_metrics_replayed'] is True
    for endpoint, filename in (('/server_info?config_format=json', 'server-info-raw.json'), ('/version', 'server-version-raw.json')):
        path = job / filename
        assert not path.exists()
        with urlopen('http://127.0.0.1:18080' + endpoint, timeout=120) as response:
            raw = response.read()
        path.write_bytes(raw)
    metadata = validate(json.loads((job / 'server-info-raw.json').read_bytes()),
                        json.loads((job / 'server-version-raw.json').read_bytes()),
                        (job / 'server.log').read_text(encoding='utf8', errors='replace'), args.mode)
    linux_job = '/mnt/d/' + job.relative_to(Path('D:/').resolve()).as_posix()
    linux_methods = '/mnt/d/CantorAI/5090-study-20261023/Garnet/test2026/cuda/qwen38_5090'
    subprocess.run(['wsl.exe', '-d', 'Ubuntu-24.04', '--', '/usr/bin/python3',
                    linux_methods + '/inspect_reference_fixed4k_linux.py', '--engine', 'vllm',
                    '--output', linux_job + '/linux-identity.json'], check=True)
    identity = json.loads((job / 'linux-identity.json').read_bytes())
    assert identity['status'] == 'reference_fixed4k_installed_sources_and_owned_parent_verified' and identity['engine'] == 'vllm'
    subprocess.run(['wsl.exe', '-d', 'Ubuntu-24.04', '--', '/usr/bin/python3',
                    linux_methods + '/inspect_vllm_fixed4k_config_linux.py', '--output', linux_job + '/vllm-config-identity.json'], check=True)
    metadata.update(launcher_sha256=sha(launcher), source_after_startup_verified=True, owned_server_pid_verified=True,
                    linux_identity_sha256=sha(job / 'linux-identity.json'), server_info_sha256=sha(job / 'server-info-raw.json'),
                    server_version_sha256=sha(job / 'server-version-raw.json'),
                    installed_config_identity_sha256=sha(job / 'vllm-config-identity.json'), native_http_proof_sha256=sha(proof_path))
    with target.open('xb') as stream:
        stream.write(json.dumps(metadata, indent=2).encode())
    print(json.dumps({'status': 'vllm_fixed4k_ready_metadata_verified', 'mode': args.mode,
                      'resolved_cache_tokens': metadata['resolved_kv_cache_tokens'], 'context_capacity': metadata['context_capacity']}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
