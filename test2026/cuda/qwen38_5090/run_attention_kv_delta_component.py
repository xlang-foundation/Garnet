"""Guarded correctness-only transactional KV-overlay component; no native build."""
from pathlib import Path
import hashlib,json,shutil,subprocess,sys
root=Path('D:/CantorAI/5090-study-20261023');methods=root/'Garnet/test2026/cuda/qwen38_5090'
sys.path.insert(0,str(methods))
from run_logged import run
sha=lambda b:hashlib.sha256(b).hexdigest()
parent='attention-b4-t4-component-attempt1';proof_path=root/'results'/(parent+'-verified.json')
proof=json.loads(proof_path.read_bytes())
assert proof['status']=='attention_b4_t4_component_replayed' and len(proof['checks'])==16
archive=root/'results'/(parent+'-evidence.zip');assert sha(archive.read_bytes())==proof['archive_sha256']
previous=json.loads((root/'results'/parent/'manifest.json').read_bytes())
assert previous['status']=='component_finished_needs_independent_replay' and previous['exit_code']==0
events=[json.loads(l) for l in (root/'results'/parent/'timing-ledger/commands.jsonl').read_text().splitlines()]
assert events[-1]['event']=='finished' and events[-1]['exit_code']==0
library=root/'build/bin/garnet.dll';assert sha(library.read_bytes())==previous['library_sha256']
prep_path=methods/'evidence/attention-kv-delta-component-preparation.json';prep=json.loads(prep_path.read_bytes())
assert sha((root/'Garnet/src/cuda/qwen35_attention_core.cu').read_bytes())==prep['native_base_sha256']
assert sha((methods/'attention_b4_t4_test.cu').read_bytes())==prep['driver_base_sha256']
for n,h in prep['files'].items():assert sha((methods/n).read_bytes())==h
for n,h in prep['method_sha256'].items():
    p=root/'build-attention-kv-delta-component.bat' if n.endswith('.bat') else methods/n
    assert sha(p.read_bytes())==h
out=root/'results/attention-kv-delta-correctness-attempt1';exe=root/'build/bin/garnet_attention_kv_delta_test.exe'
assert not out.exists() and not exe.exists()
assert int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).strip())<1024
assert shutil.disk_usage(root).free>=1024**3
out.mkdir();launcher=root/'build-attention-kv-delta-component.bat'
files=[Path(__file__),launcher,prep_path,library,proof_path,archive]+[methods/n for n in prep['files']]+[methods/'attention_b4_t4_test.cu',root/'Garnet/src/cuda/qwen35_attention_core.cu',root/'Garnet/src/cuda/qwen35_attention_core.cuh']
report={'status':'running','library_sha256':sha(library.read_bytes()),'parent_archive_sha256':proof['archive_sha256'],
        'source_sha256':{p.relative_to(root).as_posix():sha(p.read_bytes()) for p in files if p!=archive},
        'timing_enabled':False,'scope':'Standalone B4/T4 readonly KV-overlay context and explicit accepted-prefix mutable commit. Five variants, retained double CPU context/full cache witnesses and graph replays. No model integration, traffic measurement or performance claim.'}
path=out/'manifest.json';code=1
def save():path.write_text(json.dumps(report,indent=2),encoding='utf8')
save()
try:
    code=run(['cmd.exe','/d','/c',str(launcher)],out/'build-ledger','build','standalone-kv-delta-only',files)
    report['build_exit_code']=code
    if code:raise RuntimeError('Standalone component build failed')
    report['executable_sha256']=sha(exe.read_bytes())
    code=run([sys.executable,str(methods/'monitor_native_probe.py'),'--study',str(root),'--output',str(out/'resources'),'--',str(exe),str(out/'context')],out/'correctness-ledger','correctness','kv-overlay-full-prefix-gates',files+[exe])
    report['correctness_exit_code']=code
    if code:raise RuntimeError('KV overlay/prefix correctness failed')
    rows=[json.loads(l) for l in (out/'correctness-ledger/commands.log').read_text(encoding='utf8').splitlines() if l.startswith('{"case":')]
    assert len(rows)==8 and [r['case'] for r in rows]==list(range(8))
    assert all(r['status']=='correctness_pass' and r['variants_checked']==5 and r['overlay_legacy_bit_exact'] and r['readonly_overlay_preserved_kv'] and r['zero_partial_prefix_trials']==3 and r['full_kv_bit_exact'] and r['graph_replays']==3 for r in rows)
    report.update(status='correctness_finished_needs_independent_replay',rows=rows);code=0
except BaseException as failure:
    report.update(status='failed',error=repr(failure))
    if code==0:code=1
    raise
finally:
    report['exit_code']=code;report['library_unchanged']=sha(library.read_bytes())==report['library_sha256']
    report['source_after_sha256']={p.relative_to(root).as_posix():sha(p.read_bytes()) for p in files if p not in (archive,proof_path,library)}
    save()
raise SystemExit(code)
