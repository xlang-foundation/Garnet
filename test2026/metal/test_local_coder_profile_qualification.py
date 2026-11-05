"""Corrupt real archived coding/coverage evidence; never start an engine."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tempfile

from audit_local_coder_profile_qualification import audit


def read(path):
    return json.loads(path.read_text())


def write(path, data):
    path.write_text(json.dumps(data, indent=2)+'\n')


def change_events(root, change):
    path = root/'server/worker-stdout.jsonl'
    lines, deferred = [], []
    for line in path.read_text().splitlines():
        outer = json.loads(line)
        try:
            event = json.loads(outer['line'])
        except json.JSONDecodeError:
            lines.append(line)
            continue
        if isinstance(event, dict):
            action = change(event)
            if action == 'drop':
                continue
            if action == 'defer':
                deferred.append(line)
                continue
        outer['line'] = json.dumps(event)
        lines.append(json.dumps(outer))
    path.write_text('\n'.join(lines+deferred)+'\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('coding-root', 'coverage-root', 'coding-reference', 'coverage-reference', 'coding-manifest', 'suite'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    def run(coding, coverage, coding_reference=None, coverage_reference=None):
        return audit(coding, coverage, coding_reference or args.coding_reference,
                     coverage_reference or args.coverage_reference, args.coding_manifest, args.suite)
    baseline = run(args.coding_root, args.coverage_root)
    assert baseline['status'] == 'passed', baseline['issues']
    mutations = ['library', 'worker', 'response', 'prompt', 'output', 'evaluation',
                 'evaluation_response', 'evaluation_suite', 'missing_chunks',
                 'no_interleaving', 'missing_head', 'alias_layout',
                 'coding_reference', 'coverage_reference', 'profile_identity',
                 'library_identity', 'worker_identity', 'short_coverage']
    for mutation in mutations:
        with tempfile.TemporaryDirectory(prefix='garnet-profile-audit-') as folder:
            folder = Path(folder)
            coding, coverage = folder/'coding', folder/'coverage'
            shutil.copytree(args.coding_root, coding)
            shutil.copytree(args.coverage_root, coverage)
            expected = None
            coding_reference = coverage_reference = None
            if mutation in ('library', 'worker', 'library_identity', 'worker_identity'):
                identity = mutation.endswith('_identity')
                root = coverage if identity else coding
                path = root/('libgarnet.dylib' if mutation.startswith('library') else 'frozen/local_coder_batch_worker.py')
                with path.open('ab') as stream:
                    stream.write(b'\n# deliberate qualification evidence corruption\n')
                if identity:
                    data = read(root/'artifacts.json')
                    sha = hashlib.sha256(path.read_bytes()).hexdigest()
                    if mutation.startswith('library'):
                        data['library_sha256'] = sha
                        expected = 'Coding and coverage native libraries differ'
                    else:
                        data['files'][path.name] = sha
                        expected = 'Coding and coverage implementation differs: '+path.name
                    write(root/'artifacts.json', data)
            elif mutation.startswith('evaluation'):
                path = coding/'evaluation.json'
                data = read(path)
                if mutation == 'evaluation':
                    data['passed'] = 0
                elif mutation == 'evaluation_response':
                    data['results'][0]['response_sha256'] = '0'*64
                else:
                    data['suite_sha256'] = '0'*64
                write(path, data)
            elif mutation == 'response':
                name = baseline['coding_tasks'][0]['id']
                path = coding/'responses'/name/'request-0.result.json'
                data = read(path)
                data['text'] += '\npass\n'
                write(path, data)
            elif mutation in ('coding_reference', 'coverage_reference'):
                path = folder/'reference.jsonl'
                source = args.coding_reference if mutation == 'coding_reference' else args.coverage_reference
                path.write_bytes(source.read_bytes())
                fake_root = folder/'reference-root'
                (fake_root/'server').mkdir(parents=True)
                shutil.move(path, fake_root/'server/worker-stdout.jsonl')
                def alter(event):
                    if event.get('event') == 'completed':
                        event['generated_ids'][-1] += 1
                change_events(fake_root, alter)
                if mutation == 'coding_reference':
                    coding_reference = fake_root/'server/worker-stdout.jsonl'
                else:
                    coverage_reference = fake_root/'server/worker-stdout.jsonl'
                expected = ('Coding' if mutation == 'coding_reference' else 'Coverage')+': complete output token sequence differs from its frozen reference'
            elif mutation == 'profile_identity':
                artifact_path = coverage/'artifacts.json'
                artifacts = read(artifact_path)
                profiles = [coverage/'frozen'/n for n in artifacts['files'] if n.endswith('.json') and isinstance(read(coverage/'frozen'/n), dict) and 'profile_id' in read(coverage/'frozen'/n)]
                assert len(profiles) == 1
                path = profiles[0]
                data = read(path)
                data['profile_id'] += '-different'
                write(path, data)
                artifacts['files'][path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
                write(artifact_path, artifacts)
                def alter(event):
                    if event.get('event') == 'ready':
                        event['profile']['profile_id'] = data['profile_id']
                change_events(coverage, alter)
                expected = 'Coding and coverage runtime profiles differ'
            elif mutation == 'short_coverage':
                coverage = coding
                expected = 'Supplemental execution trace did not pass its full standalone audit'
            else:
                def alter(event):
                    kind = event.get('event')
                    if mutation == 'prompt' and kind == 'completed':
                        event['prompt_ids'][0] += 1
                    elif mutation == 'output' and kind == 'completed':
                        event['generated_ids'][-1] += 1
                    elif mutation == 'missing_chunks' and kind == 'native_prefill_chunk':
                        return 'drop'
                    elif mutation == 'no_interleaving' and kind == 'native_batch' and event.get('stage') == 'decode':
                        return 'defer'
                    elif mutation == 'missing_head' and kind == 'native_indexed_decode' and event.get('stage') == 'head':
                        return 'drop'
                    elif mutation == 'alias_layout' and kind == 'native_prefill_alias_prepared':
                        event['layout'][0]['groups'] += 1
                change_events(coding if mutation in ('prompt', 'output') else coverage, alter)
            result = run(coding, coverage, coding_reference, coverage_reference)
            assert result['status'] == 'failed' and result['issues'], mutation
            if expected:
                assert expected in result['issues'], (mutation, result['issues'])
            print('profile qualification rejects', mutation, flush=True)
    print('profile qualification PASS baseline and', len(mutations), 'tamper cases', flush=True)


if __name__ == '__main__':
    main()
