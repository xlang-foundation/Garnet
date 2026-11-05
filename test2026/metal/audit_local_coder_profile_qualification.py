"""Join unchanged coding contracts with separate native execution coverage.

Short coding prompts may fit in one chunk of a wider profile. Their standalone
audit remains unchanged, including any missing multi-chunk/head coverage. A
separate, fully audited trace must supply those observations using the exact
same library, worker, decoder and runtime profile. This is a bounded correctness
gate, not a performance campaign or general coding score.
"""
import argparse
import hashlib
import json
from pathlib import Path

from audit_local_coder_batch_quality import audit as trace_audit
from evaluate_local_coder_quality import extract_code


def read(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def profile(root):
    artifacts = read(root/'artifacts.json')
    profiles = []
    for name in artifacts['files']:
        if name.endswith('.json'):
            value = read(root/'frozen'/name)
            if isinstance(value, dict) and 'profile_id' in value:
                profiles.append(value)
    if len(profiles) != 1:
        raise ValueError('Exactly one frozen runtime profile required')
    return profiles[0]


def audit(coding, coverage, coding_reference, coverage_reference, coding_manifest, suite):
    issues = []
    first = trace_audit(coding, coding_reference)
    second = trace_audit(coverage, coverage_reference)
    coding_profile, coverage_profile = profile(coding), profile(coverage)
    # Only the destination of an inventory dump may differ. In particular,
    # chunk size, compiled policy, alias layout and profile identity must agree.
    for value in (coding_profile, coverage_profile):
        value.pop('memory_inventory_path', None)
    if coding_profile != coverage_profile:
        issues.append('Coding and coverage runtime profiles differ')
    artifacts = [read(root/'artifacts.json') for root in (coding, coverage)]
    if artifacts[0]['library_sha256'] != artifacts[1]['library_sha256']:
        issues.append('Coding and coverage native libraries differ')
    for name in ('local_coder_batch_worker.py', 'metal_batch_factory.py', 'metal_decoder.py'):
        if artifacts[0]['files'].get(name) != artifacts[1]['files'].get(name):
            issues.append('Coding and coverage implementation differs: '+name)
    expected_manifest = read(coding_manifest)
    frozen_manifest = coding/'responses/frozen/manifest.json'
    if digest(frozen_manifest) != digest(coding_manifest):
        issues.append('Original coding manifest differs')
    expected_tasks = [p['id'] for p in expected_manifest['plans']]
    if len(expected_tasks) != 8 or len(set(expected_tasks)) != 8:
        issues.append('Expected original eight coding contracts')
    if [t['id'] for t in first['tasks']] != expected_tasks:
        issues.append('Original coding task set differs')
    missing_coverage = {
        'Required active FP32 head projection coverage missing',
        'No existing decode advanced between native prefill chunks',
    }
    short_only = bool(first['tasks']) and all(
        t['prompt_tokens'] <= coding_profile['prefill_chunk'] for t in first['tasks'])
    transferred = [issue for issue in first['issues'] if issue in missing_coverage and short_only]
    issues.extend('Coding: '+issue for issue in first['issues'] if issue not in transferred)
    issues.extend('Coverage: '+issue for issue in second['issues'])
    if second['status'] != 'passed':
        issues.append('Supplemental execution trace did not pass its full standalone audit')
    if second.get('prefill_decode_interleavings', 0) < 1:
        issues.append('Supplemental execution lacks decode between chunks')
    if second.get('combined_decode_plans', 0) < 1:
        issues.append('Supplemental execution lacks combined decode')
    if not any(t['prompt_tokens'] > coding_profile['prefill_chunk'] for t in second['tasks']):
        issues.append('Supplemental execution lacks a multi-chunk prompt')
    for label, report in (('Coding', first), ('Coverage', second)):
        if not report['tasks'] or any(t.get('matches_single_reference') is not True for t in report['tasks']):
            issues.append(label+': complete output token sequence differs from its frozen reference')
    evaluation = read(coding/'evaluation.json')
    suite_data = read(suite)
    if (expected_manifest.get('suite_sha256') != digest(suite) or
            evaluation.get('suite_sha256') != digest(suite) or
            evaluation.get('evaluator_sha256') != digest(Path(__file__).with_name('evaluate_local_coder_quality.py'))):
        issues.append('Coding evaluation suite or evaluator differs')
    evaluated = evaluation.get('results', [])
    if (evaluation.get('passed') != 8 or evaluation.get('total') != 8 or
            evaluation.get('sandbox_probe', {}).get('status') != 'passed' or
            [r.get('task') for r in evaluated] != [t['id'] for t in suite_data['tasks']]):
        issues.append('Original sandbox coding contracts did not all pass')
    for row in evaluated:
        response = coding/'responses'/row['task']/'request-0.result.json'
        data = read(response)
        try:
            code_sha256 = hashlib.sha256(extract_code(data['text']).encode()).hexdigest()
        except (KeyError, ValueError, SyntaxError):
            code_sha256 = None
        if (row.get('status') != 'passed' or row.get('returncode') != 0 or
                row.get('response_sha256') != digest(response) or
                data.get('finish_reason') != 'stop' or row.get('finish_reason') != 'stop' or
                code_sha256 is None or row.get('code_sha256') != code_sha256):
            issues.append('Coding evaluation response/contract differs: '+row['task'])
    return dict(status='passed' if not issues else 'failed', issues=issues,
                scope=__doc__, coding_trace_status=first['status'],
                coding_trace_issues_retained=first['issues'],
                coverage_supplies=transferred, coding_tasks=first['tasks'],
                coverage_tasks=second['tasks'],
                prefill_decode_interleavings=second['prefill_decode_interleavings'],
                combined_decode_plans=second['combined_decode_plans'],
                library_sha256=artifacts[0]['library_sha256'],
                evidence_sha256={str(p): digest(p) for p in (
                    coding/'artifacts.json', coverage/'artifacts.json', coding/'evaluation.json',
                    coding/'server/worker-stdout.jsonl', coverage/'server/worker-stdout.jsonl',
                    coding_reference, coverage_reference, coding_manifest, suite)},
                auditor_sha256=digest(Path(__file__)),
                trace_auditor_sha256=digest(Path(__file__).with_name('audit_local_coder_batch_quality.py')))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('coding-root', 'coverage-root', 'coding-reference', 'coverage-reference', 'coding-manifest', 'suite', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    try:
        report = audit(args.coding_root, args.coverage_root, args.coding_reference,
                       args.coverage_reference, args.coding_manifest, args.suite)
    except (KeyError, TypeError, ValueError, OSError) as error:
        report = dict(status='failed', issues=['Malformed or missing qualification evidence: '+str(error)])
    with args.output.open('x') as stream:
        stream.write(json.dumps(report, indent=2)+'\n')
    print(json.dumps({k: report[k] for k in ('status', 'issues')}))
    return 0 if report['status'] == 'passed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
