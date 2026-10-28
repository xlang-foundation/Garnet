"""Extend an unchanged source snapshot; preserve explicitly verified completed cases."""
import argparse
import copy
import datetime
import hashlib
import json
import subprocess
import sys
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument('root', type=Path)
p.add_argument('prior_identity', type=Path)
p.add_argument('output_identity', type=Path)
a = p.parse_args()
admission_path = a.root / 'compact-kv-original-v3-reuse-admission.json'
assert not admission_path.exists() and not a.output_identity.exists()
prior = json.loads(a.prior_identity.read_text(encoding='utf-8'))
repo = Path(prior['source_root'])
# The old guard checks every registered source, controller, fixture, runtime,
# external metadata file and checkpoint stat. None is exempted or modified.
subprocess.run([sys.executable, str(repo / 'test2026/cuda/validate_qwen35_execution_identity.py'),
                str(a.prior_identity)], check=True)


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


prior_sha = digest(a.prior_identity)
log = (a.root / 'compact-kv-original-regressions-v2-controller.log').read_text(encoding='utf-8-sig')
cases = []
for terminal_path in sorted(a.root.glob('compact-kv-original-regressions-v2-*-exit.json')):
    tag = terminal_path.name[:-len('-exit.json')]
    data_path, raw_log = a.root / (tag + '.json'), a.root / (tag + '.log')
    if not data_path.exists() or not raw_log.exists():
        continue
    terminal = json.loads(terminal_path.read_text(encoding='utf-8-sig'))
    data = json.loads(data_path.read_text(encoding='utf-8-sig'))
    if terminal['exit_code'] != 0 or not data.get('all_passed'):
        continue
    assert 'COMPACT_ORIGINAL_REGRESSION_PASS ' + tag in log
    assert terminal['execution_identity_sha256'].lower() == prior_sha
    assert terminal['garnet_dll_sha256'].lower() == prior['garnet_binary_sha256'].lower()
    mode = next((name for name in ('memcheck', 'racecheck', 'synccheck') if tag.endswith('-' + name)), 'plain')
    cases.append({'tag': tag, 'mode': mode, 'execution_identity_sha256': prior_sha,
                  'files': {suffix: digest(a.root / (tag + suffix))
                            for suffix in ('.json', '.log', '-exit.json')}})
assert len(cases) == 6, 'Only the six actually completed v2 cases may be reused'
admission = {'prior_identity': a.prior_identity.name, 'prior_identity_sha256': prior_sha,
             'garnet_dll_sha256': prior['garnet_binary_sha256'], 'cases': cases,
             'all_prior_sources_inputs_and_runtime_unchanged': True,
             'verified_registered_source_count': len(prior['source_sha256']),
             'scope': 'All prior registered files pass the unchanged execution guard after the '
                      'v2 parent stopped. No model/test/runtime/input/controller exemption. Six '
                      'completed native0 cases are preserved, not rerun or relabeled as fresh. '
                      'Only new CPU orchestration/audit files are added. Remaining47 cases and '
                      'legacy smoke must still execute. No latency or memory acceptance.'}
admission_path.write_text(json.dumps(admission, indent=2), encoding='utf-8')
identity = copy.deepcopy(prior)
identity['checkpoint_full_digest_captured_at_utc'] = prior['captured_at_utc']
identity['captured_at_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
identity['resume_prior_identity_sha256'] = prior_sha
identity['resume_prior_identity_file'] = a.prior_identity.name
identity['resume_checkpoint_scope'] = 'Full checkpoint digests retained from prior capture; all saved sizes/mtimes rechecked. No new full-weight hash scan claimed.'
for name in ('prepare_qwen35_original_resume.py', 'summarize_qwen35_compact_original_regressions_v3.py'):
    relative = 'test2026/cuda/' + name
    identity['source_sha256'][relative] = digest(repo / relative)
for name in ('wait-qwen35-original-idle.ps1', 'run-compact-kv-original-resume.ps1', admission_path.name,
             a.prior_identity.name, 'compact-kv-original-regressions-v2-controller-exit.json',
             'compact-kv-original-regressions-v2-controller.log'):
    identity['orchestration_sha256'][name] = digest(a.root / name)
for case in cases:
    for suffix, sha in case['files'].items():
        identity['orchestration_sha256'][case['tag'] + suffix] = sha
a.output_identity.write_text(json.dumps(identity, indent=2), encoding='utf-8')
print(json.dumps({'unchanged_prior_source_files': len(prior['source_sha256']),
                  'completed_cases_preserved': len(cases), 'remaining_cases_to_execute': 47}))
