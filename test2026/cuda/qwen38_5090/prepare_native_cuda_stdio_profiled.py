"""Bind the actual closed unprofiled control before explicit entry-logged profiling."""
from pathlib import Path
import argparse,ast,hashlib,json,os,sys,unittest,zipfile
sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
from test2026.cuda.qwen38_5090.native_cuda_phase_study import sha

PREFIX='Garnet/test2026/cuda/qwen38_5090/'
PARENT='native-cuda-entry-profiled-bootstrap-attempt1'
NAMES=('native_profiled_worker_io.py', 'test_native_profiled_worker_io.py', 'native_cuda_phase_stdio_worker.py', 'native_cuda_phase_stdio_entry.py', 'serve_native_cuda_phase_stdio_worker.bat', 'probe_native_cuda_stdio_bootstrap.py', 'collect_native_cuda_stdio_profiled.py', 'run_native_cuda_stdio_profiled.py', 'verify_native_cuda_stdio_profiled.py', 'prepare_native_cuda_stdio_profiled.py', 'evidence/native-cuda-stdio-profiled-derivation.json')

def admit_parent(q,w,digest):
    assert q['status']=='native_cuda_entry_profiled_failure_preserved' and q['archive_sha256']==digest
    assert q['profiled'] is True and q['performance_admitted'] is q['two_times_target_achieved'] is False
    assert type(w['exit_code']) is int and w['exit_code']==1 and w['cleanup_finished'] is w['library_unchanged'] is True
    assert w['source_sha256']==w['source_after_sha256'] and w['runtime_dependencies']==w['runtime_dependencies_after']
    assert q['library_sha256']==w['library_sha256']==w['runtime_dependencies']['garnet.dll']=='1a1d0229a1c0c24a8397a4dee7a614bcedf97b3bd3293e58d752ed7de30eb498'
    assert q['control_replay'] is None
    entry=q['diagnostic_logs']['native/control/host-stages.entry.jsonl']
    error=entry[-1]['exception'];assert entry[-1]['stage']=='worker_raised' and error['type']=='OSError' and error['repr']=="OSError('write failed')"
    assert 'os.write(1,raw)' in error['traceback']
    bootstrap=q['diagnostic_logs']['native/control/host-stages.bootstrap.jsonl']
    assert 'native_ready_observed_before_pipe_emit' in [r['stage'] for r in bootstrap]

def main():
    a=argparse.ArgumentParser(description=__doc__);a.add_argument('--proofs',type=Path,required=True);a.add_argument('--work',type=Path,required=True);args=a.parse_args()
    m=Path(__file__).resolve().parent;repo=m.parents[2];path=args.proofs/(PARENT+'-verified.json');q=json.loads(path.read_bytes())
    assert path.read_bytes()==(args.proofs/(PARENT+'-remote-verified.json')).read_bytes()
    archive=args.proofs/(PARENT+'-evidence.zip');digest=sha(archive)
    with zipfile.ZipFile(archive) as z:
        manifest=json.loads(z.read('evidence-manifest.json'))
        assert z.testzip() is None and len(z.namelist())==len(set(z.namelist()))==len(manifest['files'])+1
        for n,h in manifest['files'].items():assert hashlib.sha256(z.read(n)).hexdigest()==h
        w=json.loads(z.read('results/'+PARENT+'/workflow.json'))
    admit_parent(q,w,digest)
    with zipfile.ZipFile(archive) as z:
        old={'profiler':json.loads(z.read(PREFIX+'evidence/native-cuda-entry-profiled-preparation.json'))['profiler']}
        observed=json.loads(z.read('results/'+PARENT+'/native/report.json'))
    assert observed['profiler_before']==observed['profiler_after']==old['profiler']
    files=dict(w['source_after_sha256']);new={PREFIX+n:sha(m/n) for n in NAMES};assert not(set(files)&set(new));files.update(new)
    candidate=json.loads((m/'evidence/bf16-scores-tiled-native-api-build-preparation.json').read_bytes())['candidate_files']
    for n,h in files.items():
        if n.startswith('Garnet/'):
            relative=n.removeprefix('Garnet/');assert sha(repo/relative)==candidate.get(relative,h),('Baseline/candidate source drift',n)
    for n in NAMES:
        if n.endswith('.py'):ast.parse((m/n).read_bytes())
    modules=['test2026.cuda.qwen38_5090.'+n for n in ('test_native_cuda_phases','test_native_cuda_bootstrap_log','test_native_cuda_phase_entry','test_native_cuda_entry_pipeline','test_native_profiled_worker_io')]
    assert all('Garnet/'+n.replace('.','/')+'.py' in files for n in modules)
    def ids(suite):
        for t in suite:
            if isinstance(t,unittest.TestSuite):yield from ids(t)
            else:yield t.id()
    loader=unittest.TestLoader();tests=sorted(ids(loader.loadTestsFromNames(modules)))
    assert not loader.errors and len(tests)==len(set(tests))==28
    from test2026.cuda.qwen38_5090.run_logged import run
    os.chdir(repo);code=run([sys.executable,'-m','unittest',*modules],args.work/'native-stdio-profiled-cpu-ledger','correctness',
        '28CPU-only stdio exact-frame/Unicode/partial/error/model-AST/failed-parent/preloaded-protocol and existing import/bootstrap/phase faults',[m/n for n in NAMES])
    assert code==0
    p={'status':'native_cuda_stdio_profiled_prepared_gpu_unrun','files':files,'new_files':new,'runtime_dependencies':w['runtime_dependencies_after'],
       'parent_archive_sha256':digest,'parent_proof_sha256':sha(path),'profiler':old['profiler'],
       'test_modules':modules,'test_ids':tests,'expected_cpu_tests':28,
       'scope':'One exclusive ordinary B1 actual3968+128 retained text control under pinned Nsight, after actual ready-stdout os.write failure and CPU-only stdio diagnosis. Same original request parser/Unicode normalization/ASCII JSON/partial-write semantics; only this explicit diagnostic worker uses sys.stdout text-stream callback. One frozen framed request is preloaded before readiness; delayed stdout arrivals never supply timing. Production I/O, all model/vision/MTP operations and source/runtime remain unchanged. Require28CPUcontrols and actual embedded compatibility before model, then exact IDs/host states/model close,12unique HtoD markers and owned SM120 graph-node kernels/API/copies, raw nsys/SQLite/memory gaps and independent replay. Preserve failures and owned cache retirement. Instrumented activity is not HTTP speedup or serving memory saving. Full ordinary/MTP text/image phases, matched strongest references, batched correctness/HTTP, wider prefill, repeated trials, INT4/4B and overall2x remain open.'}
    target=m/'evidence/native-cuda-stdio-profiled-preparation.json';raw=json.dumps(p,indent=2).encode()
    assert not target.exists() or target.read_bytes()==raw
    if not target.exists():target.write_bytes(raw)
    rows=[{'local':str(m/n),'target':PREFIX+n,'sha256':sha(m/n)} for n in NAMES]
    rows.append({'local':str(target),'target':PREFIX+'evidence/'+target.name,'sha256':sha(target)})
    (args.work/'native-stdio-profiled-upload-plan.json').write_bytes(json.dumps({'uploads':rows},indent=2).encode())
    print(json.dumps({'status':p['status'],'source_pins':len(files),'new_targets':len(rows),'cpu_tests':28,'preparation_sha256':sha(target)}));return 0

if __name__=='__main__':raise SystemExit(main())
