"""Run the complete frozen three-cohort plan after native MTP HTTP admission.

No comparator restart, build or download. B1 native requests queue at C2/C4.
Raw collection requires a separate complete accounting/performance audit.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
from run_local_coder_baseline import engine_processes


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    with Path(path).open('x') as stream:
        stream.write(json.dumps(value, indent=2)+'\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('experiment-root', 'admission', 'qualified', 'reference-manifest', 'asset-manifest', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--prepare-only', action='store_true', help='Validate and freeze the complete plan without starting inference')
    args = parser.parse_args()
    if engine_processes():
        raise RuntimeError('A competing inference engine is live')
    repo = Path(__file__).resolve().parents[2]
    source = Path(__file__).parent
    admission = args.admission.resolve()
    audit = json.loads((admission/'accounting-audit1.json').read_text())
    if audit['status'] != 'passed' or audit['coding_passes'] != 8 or audit['canonical_cells'] != 6 or len(audit['tasks']) != 23:
        raise ValueError('Full coding and canonical HTTP admission required')
    if digest(admission/'server/worker-stdout.jsonl') != audit['worker_log_sha256']:
        raise ValueError('Admitted worker evidence changed')
    artifacts = json.loads((admission/'artifacts.json').read_text())
    for name, wanted in artifacts['files'].items():
        if digest(admission/'sources'/name) != wanted:
            raise ValueError('Admitted implementation changed: '+name)
    qualified = json.loads((args.qualified/'qualification-provenance.json').read_text())
    archive = Path(qualified['source_archive'])
    if digest(archive/'manifest.json') != qualified['source_manifest_sha256']:
        raise ValueError('Qualified inference archive changed')
    inference = json.loads((archive/'manifest.json').read_text())
    for name, wanted in inference['sha256'].items():
        if name.startswith(('src/', 'xModel/')) and Path(name).suffix in ('.py', '.cpp', '.h', '.mm', '.json') and digest(repo/name) != wanted:
            raise ValueError('Qualified inference source changed: '+name)
    manifest = json.loads(args.reference_manifest.read_text())
    if (manifest['required_fresh_process_cohorts'] != 3 or manifest['warmup_waves_per_cell'] != 2 or
        manifest['measured_waves_per_cell'] != 5 or
        {p['id'] for p in manifest['plans']} != {f'p{p}-c{c}-o128' for p in (256,2048) for c in (1,2,4)}):
        raise ValueError('Complete unchanged three-cohort workload required')
    if digest(args.asset_manifest) != manifest['model_asset_manifest_sha256']:
        raise ValueError('Target asset manifest differs from frozen reference')
    assets = json.loads(args.asset_manifest.read_text())
    checkpoint = Path(assets['root']).resolve()
    for entry in assets['files']:
        path = checkpoint/entry['name']
        if path.stat().st_size != entry['bytes'] or digest(path) != entry['sha256']:
            raise ValueError('Pinned target bytes changed: '+entry['name'])
    original = json.loads((admission/'sources/profile.json').read_text())
    if digest(original['auxiliary_checkpoint']) != artifacts['auxiliary_sha256']:
        raise ValueError('Pinned auxiliary bytes changed')
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    frozen = root/'sources'
    shutil.copytree(admission/'sources', frozen)
    for path in [Path(__file__), source/'run_local_coder_baseline.py', source/'sample_local_coder_memory.py',
                 source/'audit_local_coder_mtp_http.py', source/'audit_local_coder_mtp_campaign.py', source/'summarize_local_coder_baseline.py']:
        shutil.copy2(path, frozen/path.name)
    shutil.copy2(args.reference_manifest, frozen/'benchmark-manifest.json')
    profiles, logs = {}, {}
    for cohort in range(1, 4):
        profile = dict(original, memory_inventory_path=str(root/f'memory-inventory-{cohort}.json'))
        name = f'profile-{cohort}.json'
        write(frozen/name, profile)
        profiles[str(cohort)] = name
        logs[str(cohort)] = f'server/cohort-{cohort}'
    original_command = json.loads((admission/'command.json').read_text())
    runtime = original_command[original_command.index('--')+1]
    command = [sys.executable, str(frozen/'local_coder_batch_http.py'), '--checkpoint', str(checkpoint),
        '--log-root', str(root/'server/cohort-{cohort}'), '--port', '8129', '--first-token-timeout', '3600', '--',
        runtime, '-B', str(frozen/'local_coder_mtp_worker.py'), str(frozen), str(repo), str(checkpoint), str(frozen/'profile-{cohort}.json')]
    spec = dict(engine='garnet-native-metal-mtp-b1-prefill128-verify4', url='http://127.0.0.1:8129', argv=command,
        env=dict(PYTHONPATH=str(frozen), GARNET_METAL_PROFILE_OPS='0'), context_capacity=4096,
        max_running_requests=1, queued_concurrency=True, native_continuous_batching=False,
        prefix_caching=False, enable_multimodal=False, prefill_chunk=128,
        auxiliary_draft_chunk=1, verification_chunk=4, target_asset_manifest_sha256=digest(args.asset_manifest),
        auxiliary_asset_sha256=artifacts['auxiliary_sha256'], qualified_library_sha256=artifacts['qualified_library_sha256'],
        precision_scope='Pinned target affine INT4/group64 with FP32 target activations/KV; shared raw BF16 auxiliary weights, FP32 accumulation; exact target verification',
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'], cwd=repo, text=True).strip(),
        sources={str(p.relative_to(frozen)):digest(p) for p in frozen.rglob('*') if p.is_file()},
        native_cohort_profiles=profiles, native_cohort_log_roots=logs, graceful_native_shutdown=True,
        request_timeout_seconds=3600, startup_observation_seconds=600,
        reference_manifest_sha256=digest(args.reference_manifest),
        admitted_http_archive=str(admission), admission_audit_sha256=digest(admission/'accounting-audit1.json'),
        qualified_model_archive=str(args.qualified.resolve()), qualified_source_manifest_sha256=qualified['source_manifest_sha256'],
        scope=__doc__)
    write(root/'engine.json', spec)
    write(root/'asset-verification.json', dict(status='passed', model_asset_manifest_sha256=digest(args.asset_manifest),
        files=len(assets['files']), bytes=sum(e['bytes'] for e in assets['files']), auxiliary_sha256=artifacts['auxiliary_sha256']))
    driver = [sys.executable, str(frozen/'run_local_coder_baseline.py'), '--experiment-root', str(args.experiment_root.resolve()),
        '--manifest', str(frozen/'benchmark-manifest.json'), '--engine-spec', str(root/'engine.json'), '--output', str(root/'campaign'), '--cohorts', '3']
    write(root/'command.json', driver)
    if args.prepare_only:
        write(root/'preparation.json', dict(status='prepared_not_started', scope='Complete frozen plan validation only; no fresh GPU cohorts or performance evidence'))
        print('prepared complete frozen native MTP campaign; inference not started', flush=True)
        return
    print('starting complete frozen native MTP campaign', flush=True)
    with (root/'driver.log').open('x') as log:
        subprocess.run(driver, stdout=log, stderr=subprocess.STDOUT, check=True)
    for name, wanted in spec['sources'].items():
        if digest(frozen/name) != wanted:
            raise ValueError('Measured source changed: '+name)
    for name, wanted in inference['sha256'].items():
        if name.startswith(('src/', 'xModel/')) and Path(name).suffix in ('.py', '.cpp', '.h', '.mm', '.json') and digest(repo/name) != wanted:
            raise ValueError('Measured inference source changed: '+name)
    write(root/'post-run-source-verification.json', dict(status='passed',
        engine_spec_sha256=digest(root/'engine.json'), qualified_source_manifest_sha256=qualified['source_manifest_sha256'],
        measured_sources=spec['sources'], scope='All frozen files and qualified inference sources unchanged after all three cohorts'))
    print('completed collection; full campaign audit still required', flush=True)


if __name__ == '__main__':
    main()
