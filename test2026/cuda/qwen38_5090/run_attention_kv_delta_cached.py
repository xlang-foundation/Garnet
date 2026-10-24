"""Qualify cached/hybrid KV overlays before separate component event timing."""
from pathlib import Path
import hashlib,json,math,shutil,subprocess,sys
root=Path('D:/CantorAI/5090-study-20261023');methods=root/'Garnet/test2026/cuda/qwen38_5090'
sys.path.insert(0,str(methods))
from run_logged import run
sha=lambda b:hashlib.sha256(b).hexdigest()
parent='attention-kv-delta-timing-attempt1';proof_path=root/'results'/(parent+'-verified.json')
proof=json.loads(proof_path.read_bytes())
assert proof['status']=='attention_kv_delta_timing_replayed' and len(proof['cases'])==16 and proof['raw_context_arrays']==96 and proof['raw_prefix_count_vectors']==48
archive=root/'results'/(parent+'-evidence.zip');assert sha(archive.read_bytes())==proof['archive_sha256']
previous=json.loads((root/'results'/parent/'manifest.json').read_bytes())
assert previous['status']=='component_timing_finished_needs_independent_replay' and previous['exit_code']==0
events=[json.loads(l) for l in (root/'results'/parent/'timing-ledger/commands.jsonl').read_text().splitlines()]
assert events[-1]['event']=='finished' and events[-1]['exit_code']==0
library=root/'build/bin/garnet.dll';assert sha(library.read_bytes())==previous['library_sha256']
prep_path=methods/'evidence/attention-kv-delta-cached-preparation.json';prep=json.loads(prep_path.read_bytes())
for group in ('base_source_sha256','unchanged_source_sha256'):
    for n,h in prep[group].items():assert sha((root/'Garnet'/n).read_bytes())==h
for n,h in prep['files'].items():assert sha((methods/n).read_bytes())==h
for n,h in prep['method_sha256'].items():
    path=root/'build-attention-kv-delta-cached.bat' if n.endswith('.bat') else methods/n
    assert sha(path.read_bytes())==h
driver=(methods/'attention_kv_delta_timing_test.cu').read_bytes().decode()
for old,new in prep['driver_derivation']:
    assert driver.count(old)==1;driver=driver.replace(old,new)
assert driver.count('mode==4?inputs[')==prep['canonical_cache_comparison_replacement_count']==6
driver=driver.replace('mode==4?inputs[','mode>=4?inputs[')
assert driver.encode()==(methods/'attention_kv_delta_cached_test.cu').read_bytes()
out=root/'results/attention-kv-delta-cached-attempt1';exe=root/'build/bin/garnet_attention_kv_delta_cached_test.exe'
assert not out.exists() and not exe.exists()
assert int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).strip())<1024
assert shutil.disk_usage(root).free>=1024**3
out.mkdir();launcher=root/'build-attention-kv-delta-cached.bat'
files=[Path(__file__),launcher,prep_path,library,proof_path,archive]+[methods/n for n in prep['files']]+[root/'Garnet'/n for group in ('base_source_sha256','unchanged_source_sha256') for n in prep[group]]
report={'status':'running','library_sha256':sha(library.read_bytes()),'parent_archive_sha256':proof['archive_sha256'],
        'source_sha256':{p.relative_to(root).as_posix():sha(p.read_bytes()) for p in files if p!=archive},'hybrid_threshold':512,
        'scope':'Standalone twelve-profile/eight-strategy component; cached slots, compact versus legacy geometry, per-query hybrid512. Readonly overlays plus same accepted-prefix commit; different ownership/output contract from full-cache legacy. No model/HTTP/DRAM/2x evidence.'}
path=out/'manifest.json';code=1
def save():path.write_text(json.dumps(report,indent=2),encoding='utf8')
save()
try:
    code=run(['cmd.exe','/d','/c',str(launcher)],out/'build-ledger','build','standalone-cached-kv-overlay',files)
    report['build_exit_code']=code
    if code:raise RuntimeError('Cached component build failed')
    report['executable_sha256']=sha(exe.read_bytes())
    for stage,extra in [('qualification',[]),('timing',['--timing'])]:
        code=run([sys.executable,str(methods/'monitor_native_probe.py'),'--study',str(root),'--output',str(out/(stage+'-resources')),'--',str(exe),str(out/(stage+'-context'))]+extra,out/(stage+'-ledger'),'correctness','cached-kv-'+stage,files+[exe])
        report[stage+'_exit_code']=code
        if code:raise RuntimeError(stage+' failed')
    rows=[json.loads(l) for l in (out/'timing-ledger/commands.log').read_text(encoding='utf8').splitlines() if l.startswith('{"case":')]
    assert len(rows)==12 and [r['case'] for r in rows]==list(range(12))
    for row in rows:
        assert row['order']=='ABCDEFGHHGFEDCBAx3' and row['iterations_per_sample']==20 and row['warmups_per_variant']==5
        assert row['overlay_legacy_bit_exact'] and row['timed_context_exact'] and row['timed_full_kv_exact']
        for key in ('baseline_ms','compact_ms','tiled_ms','split_ms','overlay_commit_ms','cached_compact_commit_ms','cached_legacy_commit_ms','hybrid_commit_ms'):
            assert len(row[key])==6 and all(math.isfinite(v) and v>0 for v in row[key])
    report.update(status='cached_component_finished_needs_independent_replay',rows=rows);code=0
except BaseException as failure:
    report.update(status='failed',error=repr(failure))
    if code==0:code=1
    raise
finally:
    report['exit_code']=code;report['library_unchanged']=sha(library.read_bytes())==report['library_sha256']
    report['source_after_sha256']={p.relative_to(root).as_posix():sha(p.read_bytes()) for p in files if p not in (archive,proof_path,library)}
    save()
raise SystemExit(code)
