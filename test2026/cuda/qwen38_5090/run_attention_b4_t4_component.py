"""Separate B4/T4 attention component qualification after layer diagnostic closure."""
from pathlib import Path
import hashlib, json, math, subprocess, sys

root = Path('D:/CantorAI/5090-study-20261023')
methods = root/'Garnet/test2026/cuda/qwen38_5090'
sys.path.insert(0, str(methods))
from run_logged import run

sha = lambda b: hashlib.sha256(b).hexdigest()
parent = 'b4-joined-layer-profile-attempt1'
proof_path = root/'results'/(parent+'-verified.json')
proof = json.loads(proof_path.read_bytes())
assert proof['status'] == 'b4_joined_layer_profile_replayed' and proof['correctness_passed']
archive = root/'results'/(parent+'-evidence.zip')
assert sha(archive.read_bytes()) == proof['archive_sha256']
workflow = json.loads((root/'results'/parent/'workflow.json').read_bytes())
events = [json.loads(l) for l in (root/'results'/parent/'ledger/commands.jsonl').read_text().splitlines()]
assert events[-1]['event'] == 'finished' and events[-1]['exit_code'] == workflow['exit_code'] == 0
library = root/'build/bin/garnet.dll'
assert sha(library.read_bytes()) == workflow['library_sha256']
for name, want in workflow['workspace_source_sha256'].items():
    assert sha((root/'Garnet'/name).read_bytes()) == want
prep_path = methods/'evidence/b4-t4-attention-component-preparation.json'
prep = json.loads(prep_path.read_bytes())
for name, want in prep['native_source_sha256'].items():
    assert sha((root/'Garnet'/name).read_bytes()) == want
for name, want in prep['method_sha256'].items():
    path = root/'build-attention-b4-t4-component.bat' if name.endswith('.bat') else methods/name
    assert sha(path.read_bytes()) == want
original = (root/'Garnet'/prep['base_file']).read_bytes()
assert sha(original) == prep['base_sha256']
derived = original.decode().replace('\r\n', '\n')
for change in prep['derivation']:
    assert derived.count(change['old']) == 1
    derived = derived.replace(change['old'], change['new'])
candidate = root/'Garnet'/prep['candidate_file']
assert candidate.read_bytes() == derived.encode()
assert sha(candidate.read_bytes()) == prep['candidate_sha256']
out = root/'results/attention-b4-t4-component-attempt1'
exe = root/'build/bin/garnet_attention_b4_t4_test.exe'
assert not out.exists() and not exe.exists()
assert int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).strip()) < 1024
out.mkdir()
launcher = root/'build-attention-b4-t4-component.bat'
core = [root/'Garnet/src/cuda'/n for n in ('qwen35_attention_core.cu','qwen35_attention_core.cuh')]
files = [Path(__file__), launcher, candidate, root/'Garnet'/prep['base_file'], prep_path,
         proof_path, archive, library] + core
report = {'status':'running','library_sha256':sha(library.read_bytes()),
          'parent_archive_sha256':proof['archive_sha256'],
          'source_sha256':{p.relative_to(root).as_posix():sha(p.read_bytes()) for p in files if p != archive},
          'scope':'Standalone synthetic B4/T4 FP32 attention with full KV copies. Existing kernels unchanged; independent double CPU oracle, cache/input/guard/graph checks precede CUDA-event timing. Not full-model correctness, actual4K prompts, serving latency or2x evidence.'}
manifest = out/'manifest.json'
def save(): manifest.write_text(json.dumps(report,indent=2),encoding='utf8')
save()
code = 1
try:
    code = run(['cmd.exe','/d','/c',str(launcher)],out/'build-ledger','build','b4-t4-component-only',files)
    report['build_exit_code'] = code
    if code: raise RuntimeError('Component build failed')
    report['executable_sha256'] = sha(exe.read_bytes())
    for stage, extra in [('qualification',[]),('timing',['--timing'])]:
        code = run([sys.executable,str(methods/'monitor_native_probe.py'),'--study',str(root),
                    '--output',str(out/(stage+'-resources')),'--',str(exe),str(out/(stage+'-context'))]+extra,
                   out/(stage+'-ledger'),'correctness','b4-t4-attention-'+stage,files+[exe])
        report[stage+'_exit_code'] = code
        if code: raise RuntimeError(stage+' failed')
    raw = (out/'timing-ledger/commands.log').read_text(encoding='utf8')
    rows = [json.loads(line) for line in raw.splitlines() if line.startswith('{"case":')]
    assert len(rows) == 8 and [r['case'] for r in rows] == list(range(8))
    for row in rows:
        assert row['batch']==4 and row['tokens']==4 and row['capacity']==4096
        assert row['order']=='ABCDDCBAx3' and row['row_tiled_bit_exact'] and row['split_within_tolerance'] and row['full_kv_exact']
        for key in ('baseline_ms','compact_ms','tiled_ms','split_ms'):
            assert len(row[key])==6 and all(math.isfinite(v) and v>0 for v in row[key])
    report.update(status='component_finished_needs_independent_replay',rows=rows)
    code = 0
except BaseException as error:
    report.update(status='failed',error=repr(error))
    if code==0: code=1
    raise
finally:
    report['exit_code']=code
    report['library_unchanged']=sha(library.read_bytes())==report['library_sha256']
    report['source_after_sha256']={p.relative_to(root).as_posix():sha(p.read_bytes()) for p in core+[candidate]}
    save()
raise SystemExit(code)
