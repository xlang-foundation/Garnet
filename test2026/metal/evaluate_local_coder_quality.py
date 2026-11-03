"""Evaluate saved Python completions in a bounded macOS sandbox subprocess.

Never imports model packages or starts an inference engine. Infrastructure
errors are distinct from model failures. All supplied tasks receive a result;
length-truncated completions cannot establish a successful quality gate.
"""
import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import tempfile
import time


def extract_code(text):
    text = text.strip()
    if text.startswith('```'):
        match = re.fullmatch(r'```(?:python|py)?\s*\n(.*?)\n```', text, re.S)
        if not match:
            raise ValueError('Expected a single complete Python code block')
        text = match.group(1)
    ast.parse(text)
    return text


def sandbox_profile(python, directory):
    # Use the resolved interpreter with -I -S: no user packages, user site,
    # PYTHONPATH or external engine runtime imports. Model output can only
    # write inside the disposable directory and cannot open network sockets.
    runtime = str(python.parent.parent)
    quote = json.dumps
    return '\n'.join([
        '(version 1)', '(deny default)', '(allow process-exec)',
        '(allow file-read-metadata)',
        # dyld's ignition/cache discovery reads the root directory. Literal
        # access permits that directory only, not arbitrary descendant data.
        '(allow file-read* (literal "/"))',
        '(allow file-map-executable (subpath "/System") (subpath "/usr/lib") '
        f'(subpath {quote(runtime)}))',
        '(allow file-read* (subpath "/System") (subpath "/usr/lib") '
        '(subpath "/private/var/db/dyld") '
        f'(subpath {quote(runtime)}) (subpath {quote(str(directory))}))',
        f'(allow file-write* (subpath {quote(str(directory))}) (literal "/dev/null"))',
        '(allow sysctl-read)', '(allow mach-lookup)',
    ])+'\n'


def execute(code, tests, seconds=10):
    sandbox = Path('/usr/bin/sandbox-exec')
    if sys.platform != 'darwin' or not sandbox.exists():
        return dict(status='infrastructure_error', error='Required macOS sandbox-exec unavailable')
    python = Path(sys.executable).resolve()
    with tempfile.TemporaryDirectory(prefix='garnet-quality-') as temporary:
        folder = Path(temporary).resolve()
        profile = folder/'policy.sb'
        profile.write_text(sandbox_profile(python, folder))
        source = folder/'check.py'
        marker = 'GARNET_QUALITY_PASS:'+os.urandom(24).hex()
        source.write_text(
            'import resource, sys, traceback\n'
            'def limit(kind, requested):\n'
            '    soft, hard = resource.getrlimit(kind)\n'
            '    bounded = requested if hard == resource.RLIM_INFINITY else min(requested, hard)\n'
            '    resource.setrlimit(kind, (bounded, bounded))\n'
            'limit(resource.RLIMIT_CPU, 5)\n'
            'limit(resource.RLIMIT_FSIZE, 1048576)\n'
            'limit(resource.RLIMIT_NOFILE, 32)\n'
            'resource.setrlimit(resource.RLIMIT_CORE, (0, 0))\n'
            "namespace = {'__name__': 'candidate'}\n"
            'try:\n'
            '    exec(compile('+repr(code)+", '<candidate>', 'exec'), namespace)\n"
            '    exec(compile('+repr(tests)+", '<contracts>', 'exec'), namespace)\n"
            'except BaseException:\n'
            '    traceback.print_exc()\n'
            '    sys.exit(1)\n'
            'print('+repr(marker)+', flush=True)\n')
        environment = dict(PATH='/usr/bin:/bin', HOME=str(folder), LANG='C', PYTHONHASHSEED='0')
        with (folder/'stdout').open('wb') as out, (folder/'stderr').open('wb') as err:
            process = subprocess.Popen([str(sandbox), '-f', str(profile), str(python),
                                        '-I', '-S', str(source)], cwd=folder, env=environment,
                                       stdout=out, stderr=err, start_new_session=True)
            timed_out = memory_exceeded = False
            deadline = time.monotonic()+seconds
            while process.poll() is None:
                usage = subprocess.run(['/bin/ps', '-p', str(process.pid), '-o', 'rss='],
                                       capture_output=True, text=True).stdout.strip()
                if process.poll() is not None:
                    break
                memory_exceeded = bool(usage and int(usage) > 512*1024)
                timed_out = time.monotonic() >= deadline
                if not memory_exceeded and not timed_out:
                    time.sleep(.05)
                    continue
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait(timeout=5)
                break
        raw_stdout = (folder/'stdout').read_bytes()
        stdout = raw_stdout[:65536].decode(errors='replace')
        stderr = (folder/'stderr').read_bytes()[:65536].decode(errors='replace')
        passed = process.returncode == 0 and marker.encode() in raw_stdout.splitlines()
        status = 'memory_limit' if memory_exceeded else ('timeout' if timed_out else ('passed' if passed else 'failed'))
        if 'sandbox-exec:' in stderr or 'sandbox_init:' in stderr:
            status = 'infrastructure_error'
        return dict(status=status, returncode=process.returncode, stdout=stdout, stderr=stderr)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--suite', type=Path, required=True)
    parser.add_argument('--responses', type=Path, required=True,
                        help='Directory with one task-id/request-0.result.json per frozen task')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    suite = json.loads(args.suite.read_text())
    # Prove sandbox startup separately; do not call every model wrong when the
    # OS rejects the evaluator. The containment self-test is a separate gate.
    probe = execute('value = 6*7', 'assert value == 42')
    results = []
    for task in suite['tasks']:
        path = args.responses/task['id']/'request-0.result.json'
        row = dict(task=task['id'], status='missing_response')
        if path.exists():
            response = json.loads(path.read_text())
            row.update(response_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                       finish_reason=response.get('finish_reason'))
            if probe['status'] != 'passed':
                row.update(status='infrastructure_error', error='Sandbox startup probe failed')
            elif response.get('status') != 'completed':
                row['status'] = 'generation_failed'
            elif response.get('finish_reason') == 'length':
                row['status'] = 'truncated'
            else:
                try:
                    code = extract_code(response['text'])
                    row.update(execute(code, task['tests']))
                    row['code_sha256'] = hashlib.sha256(code.encode()).hexdigest()
                except (ValueError, SyntaxError, KeyError) as error:
                    row.update(status='invalid_code', error=str(error))
        results.append(row)
    report = dict(scope=suite['scope'], suite_sha256=hashlib.sha256(args.suite.read_bytes()).hexdigest(),
                  evaluator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  sandbox_probe=probe, results=results,
                  passed=sum(row['status'] == 'passed' for row in results), total=len(results),
                  interpretation='Bounded regression contracts only; not a general coding leaderboard score')
    with args.output.open('x') as stream:
        json.dump(report, stream, indent=2)
        stream.write('\n')
    print(json.dumps(dict(passed=report['passed'], total=report['total'], sandbox_probe=probe['status'])))


if __name__ == '__main__':
    main()
