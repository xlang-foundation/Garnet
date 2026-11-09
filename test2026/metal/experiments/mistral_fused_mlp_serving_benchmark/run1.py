"""Preflight once, collect four serial native waves, then postflight once.

No comparator execution. No model/source hashing between timed waves.
Every child is waited to termination; failures stop the campaign without retry.
"""
from pathlib import Path
import hashlib, json, os, subprocess, sys, time

ROOT = Path(__file__).resolve().parent
M = ROOT.parent
WORKSPACE = M.parents[2]
load = lambda p: json.loads(p.read_text())

def sha(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def save(p, value):
    temporary = p.with_name(p.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(p)

def main():
    assert not (ROOT / 'collection-status1.json').exists(), 'Never reuse a campaign'
    sys.path.insert(0, str(ROOT / 'tools1'))
    from run_local_coder_baseline import engine_processes
    assert not engine_processes(), engine_processes()
    checked = {}
    def bind(p, expected=None):
        p = Path(p).resolve()
        value = sha(p)
        if expected is not None:
            assert value == expected, ('changed bound file', str(p))
        if str(p) in checked:
            assert checked[str(p)] == value
        checked[str(p)] = value
        return value
    gates = [('mistral-fused-mlp-native1', 'provenance1.json', 'b94f6e7ee34d17079439e88cd0d5e419ff9c11e8d2514fbe2dbd00f589917ebe'), ('mistral-fused-mlp-prefix-http1', 'provenance1.json', 'd9abf05f94f2c8f92c066717eee3ca5824ed39766c7412e8923d0bf2ab4ed96c'), ('mistral-fused-mlp-prefix-pressure1', 'provenance1.json', '6ce2c8ea48493a08347e3e340716800e10ba8f099c482a4bd8864eb90f0066d2'), ('mistral-fused-mlp-coding-quality1', 'provenance1.json', '6ac91f90b23d98f2bf82dacc3a99747e0f50f85f6a28cdbdb87a5eae1dee806f'), ('mistral-fused-mlp-native-replay-preparation1', 'provenance1.json', '70f271b2b1f97c99a09693b8cf37ba13cfab742d46cd832a6e03344e9063c70d')]
    for folder, name, digest in gates:
        p = M / folder / name
        bind(p, digest)
        for relative, expected in load(p)['files_sha256'].items():
            bind(p.parent / relative, expected)
    quality = load(M / 'mistral-fused-mlp-coding-quality1/fused-quality-audit1.json')
    assert quality['passed'] and quality['quality_gate_passed'] is False
    native = M / 'mistral-half-attention-native1'
    audit = load(native / 'source-audit7.json')
    for relative, digest in audit['original_dirty_sha256'].items():
        bind(WORKSPACE / 'Garnet' / relative, digest)
    source = M / 'mistral-fused-mlp-native1/source2'
    portable = M / 'mistral-fused-mlp-native-replay-preparation1/source2'
    source_files = {}
    for p in sorted(source.rglob('*')):
        if p.is_symlink():
            relative = str(p.relative_to(source))
            assert (portable / relative).is_symlink()
            assert os.readlink(p) == os.readlink(portable / relative)
            source_files[relative] = 'symlink:' + os.readlink(p)
        elif p.is_file():
            relative = str(p.relative_to(source))
            value = bind(p)
            bind(portable / relative, value)
            source_files[relative] = value
    assert len(source_files) == 26294, len(source_files)
    for relative, digest in load(M / 'mistral-fused-mlp-native1/source-bindings2.json')['files_sha256'].items():
        assert source_files[relative] == digest
    reference_path = M / 'mistral-repeated-reference-provenance1.json'
    bind(reference_path)
    reference = load(reference_path)
    assert all(e['completed_cohorts'] == 3 for e in reference['engines'])
    for path, digest in reference['files'].items():
        bind(path, digest)
    plan = load(ROOT / 'plan1.json')
    model = Path(plan['model'])
    for relative, digest in plan['model_files_sha256'].items():
        bind(model / relative, digest)
    bind(Path(plan['library']) / 'libgarnet.dylib', plan['library_sha256'])
    bind(plan['native_model_binding'], plan['native_model_binding_sha256'])
    binding = load(Path(plan['native_model_binding']))
    bind(binding['xlang'], binding['xlang_sha256'])
    bind(ROOT / 'tools1/local_coder_batch_worker.py', plan['unprofiled_worker_sha256'])
    for p in (ROOT / 'tools1').glob('*.py'):
        compile(p.read_text(), str(p), 'exec')
        bind(p)
    driver = (ROOT / 'tools1/run_mistral_cold_prefix.py').read_text()
    assert 'threading.Barrier(concurrency)' in driver and 'max_workers=concurrency' in driver
    assert 'hashlib.sha256(' not in driver
    for p in [ROOT / 'prepare1.py', ROOT / 'run1.py', ROOT / 'plan1.json', ROOT / 'serving-manifest1.json', ROOT / 'fixture1.json', ROOT / 'audit1.py', ROOT / 'preparation1.json']:
        bind(p)
    for p in (ROOT / 'audit-tools1').iterdir():
        if p.is_file(): bind(p)
    for case in load(ROOT / 'serving-manifest1.json')['plans']:
        bind(case['cases'], case['cases_sha256'])
    for wave in plan['waves']:
        folder = ROOT / wave['label']
        bind(folder / 'engine.json', wave['engine_sha256'])
        bind(folder / 'profile.json', wave['profile_sha256'])
        current_profile = load(folder / 'profile.json')
        admitted_profile = load(M / 'mistral-fused-mlp-coding-quality1/profile1.json')
        allowed_changes = {'profile_id', 'scope', 'prefix_identity', 'prefix_entries', 'prefix_host_bytes', 'memory_inventory_path'}
        assert {k: v for k, v in current_profile.items() if k not in allowed_changes} == {k: v for k, v in admitted_profile.items() if k not in allowed_changes}
        assert current_profile['prefix_capture_policy'] == 'prefill_only_v1'
        assert current_profile['prefix_entries'] == current_profile['prefix_host_bytes'] == 0
        spec = load(folder / 'engine.json')
        bind(spec['saved_native_control'])
        assert not (folder / 'campaign').exists()
    assert not engine_processes(), engine_processes()
    save(ROOT / 'preflight1.json', dict(passed=True, finished_ns=time.time_ns(), files_sha256=checked,
         source_files=source_files, quality_gate_passed=False, no_comparator_execution=True,
         normal_request_body_sha_in_timed_client=True,
         request_hash_scope='Original transport hashes the small request body after submitted clock; unchanged for all native/reference clients. No added source/model hashing during campaign.'))
    state = dict(status='collecting', pid=os.getpid(), started_ns=time.time_ns(), waves=[])
    save(ROOT / 'collection-status1.json', state)
    print(json.dumps(dict(preflight=True, files=len(checked), pid=os.getpid())), flush=True)
    try:
        for wave in plan['waves']:
            assert not engine_processes(), engine_processes()
            row = dict(label=wave['label'], case=wave['case'], status='starting', started_ns=time.time_ns(), command=wave['command'])
            state['waves'].append(row)
            save(ROOT / 'collection-status1.json', state)
            with (ROOT / wave['label'] / 'collector.log').open('x') as log:
                child = subprocess.Popen(wave['command'], stdout=log, stderr=subprocess.STDOUT)
                row.update(pid=child.pid, status='running')
                save(ROOT / 'collection-status1.json', state)
                print(json.dumps(dict(started=wave['label'], pid=child.pid)), flush=True)
                code = child.wait()
            row.update(returncode=code, finished_ns=time.time_ns(), status='collected' if code == 0 else 'failed')
            save(ROOT / 'collection-status1.json', state)
            assert code == 0, row
            assert not engine_processes(), engine_processes()
            print(json.dumps(dict(terminal=wave['label'], returncode=code)), flush=True)
        state.update(status='collected_pending_independent_audit', finished_ns=time.time_ns())
    except BaseException as error:
        state.update(status='failed', error=repr(error), finished_ns=time.time_ns())
        raise
    finally:
        save(ROOT / 'collection-status1.json', state)
    # All four subprocesses and inference groups are terminal before hashing.
    assert not engine_processes(), engine_processes()
    for path, digest in checked.items():
        assert sha(Path(path)) == digest, ('changed after campaign', path)
    save(ROOT / 'postflight1.json', dict(passed=True, finished_ns=time.time_ns(), checked_files=len(checked), all_engines_terminal=True))
    print(json.dumps(dict(status=state['status'], postflight=True)), flush=True)

if __name__ == '__main__':
    main()
