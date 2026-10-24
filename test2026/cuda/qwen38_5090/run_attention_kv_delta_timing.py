"""Qualify a separate timing binary, then record internal component CUDA events."""
from pathlib import Path
import hashlib,json,math,shutil,subprocess,sys
root=Path('D:/CantorAI/5090-study-20261023');methods=root/'Garnet/test2026/cuda/qwen38_5090'
sys.path.insert(0,str(methods))
from run_logged import run
sha=lambda b:hashlib.sha256(b).hexdigest()
parent='attention-kv-delta-correctness-attempt1';proof_path=root/'results'/(parent+'-verified.json')
proof=json.loads(proof_path.read_bytes())
assert proof['status']=='attention_kv_delta_component_replayed' and len(proof['cases'])==8 and proof['raw_context_arrays']==48 and proof['raw_prefix_count_vectors']==24
archive=root/'results'/(parent+'-evidence.zip');assert sha(archive.read_bytes())==proof['archive_sha256']
previous=json.loads((root/'results'/parent/'manifest.json').read_bytes())
assert previous['status']=='correctness_finished_needs_independent_replay' and previous['exit_code']==0
events=[json.loads(l) for l in (root/'results'/parent/'correctness-ledger/commands.jsonl').read_text().splitlines()]
assert events[-1]['event']=='finished' and events[-1]['exit_code']==0
library=root/'build/bin/garnet.dll';assert sha(library.read_bytes())==previous['library_sha256']
prep_path=methods/'evidence/attention-kv-delta-timing-preparation.json';prep=json.loads(prep_path.read_bytes())
for n,h in prep['native_source_sha256'].items():assert sha((root/'Garnet'/n).read_bytes())==h
for n,h in prep['unchanged_source_sha256'].items():assert sha((methods/n).read_bytes())==h
base=(methods/prep['base_driver']).read_bytes();assert sha(base)==prep['base_driver_sha256']
derived=base.decode()
for old,new in prep['driver_derivation']:
    assert derived.count(old)==1;derived=derived.replace(old,new)
marker='\n }\n Check(cudaEventDestroy';assert derived.count(marker)==1
derived=derived.replace(marker,'\n'+prep['timing_block']+marker)
driver=methods/prep['driver'];assert driver.read_bytes()==derived.encode() and sha(driver.read_bytes())==prep['driver_sha256']
for n,h in prep['method_sha256'].items():
    path=root/'build-attention-kv-delta-timing.bat' if n.endswith('.bat') else methods/n
    assert sha(path.read_bytes())==h
out=root/'results/attention-kv-delta-timing-attempt1';exe=root/'build/bin/garnet_attention_kv_delta_timing_test.exe'
assert not out.exists() and not exe.exists()
assert int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).strip())<1024
assert shutil.disk_usage(root).free>=1024**3
out.mkdir();launcher=root/'build-attention-kv-delta-timing.bat'
files=[Path(__file__),launcher,prep_path,library,proof_path,archive,driver,methods/prep['base_driver']]+[methods/n for n in prep['unchanged_source_sha256']]+[root/'Garnet'/n for n in prep['native_source_sha256']]
report={'status':'running','library_sha256':sha(library.read_bytes()),'parent_archive_sha256':proof['archive_sha256'],
        'source_sha256':{p.relative_to(root).as_posix():sha(p.read_bytes()) for p in files if p!=archive},'timing_enabled':True,
        'scope':'Standalone frozen-slot CUDA-event component after same-binary qualification. Legacy full-cache outputs versus readonly KV overlay plus in-place full accepted-prefix commit; changed ownership/output contract. Common initialized canonical caches; readbacks/checks outside events. Not model/HTTP/DRAM-traffic/2x evidence.'}
path=out/'manifest.json';code=1
def save():path.write_text(json.dumps(report,indent=2),encoding='utf8')
save()
try:
    code=run(['cmd.exe','/d','/c',str(launcher)],out/'build-ledger','build','standalone-kv-delta-timing-binary',files)
    report['build_exit_code']=code
    if code:raise RuntimeError('Timing binary build failed')
    report['executable_sha256']=sha(exe.read_bytes())
    for stage,extra in [('qualification',[]),('timing',['--timing'])]:
        code=run([sys.executable,str(methods/'monitor_native_probe.py'),'--study',str(root),'--output',str(out/(stage+'-resources')),'--',str(exe),str(out/(stage+'-context'))]+extra,out/(stage+'-ledger'),'correctness','kv-delta-'+stage,files+[exe])
        report[stage+'_exit_code']=code
        if code:raise RuntimeError(stage+' failed')
    rows=[json.loads(l) for l in (out/'timing-ledger/commands.log').read_text(encoding='utf8').splitlines() if l.startswith('{"case":')]
    assert len(rows)==8 and [r['case'] for r in rows]==list(range(8))
    for row in rows:
        assert row['order']=='ABCDEEDCBAx3' and row['iterations_per_sample']==20 and row['warmups_per_variant']==5
        assert row['overlay_legacy_bit_exact'] and row['timed_context_exact'] and row['timed_full_kv_exact']
        for key in ('baseline_ms','compact_ms','tiled_ms','split_ms','overlay_commit_ms'):
            assert len(row[key])==6 and all(math.isfinite(v) and v>0 for v in row[key])
    report.update(status='component_timing_finished_needs_independent_replay',rows=rows);code=0
except BaseException as failure:
    report.update(status='failed',error=repr(failure))
    if code==0:code=1
    raise
finally:
    report['exit_code']=code;report['library_unchanged']=sha(library.read_bytes())==report['library_sha256']
    report['source_after_sha256']={p.relative_to(root).as_posix():sha(p.read_bytes()) for p in files if p not in (archive,proof_path,library)}
    save()
raise SystemExit(code)
