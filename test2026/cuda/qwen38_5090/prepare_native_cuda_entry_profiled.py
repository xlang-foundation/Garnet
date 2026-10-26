"""Bind the actual closed unprofiled control before explicit entry-logged profiling."""
from pathlib import Path
import argparse,ast,hashlib,json,os,sys,unittest,zipfile
sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
from test2026.cuda.qwen38_5090.native_cuda_phase_study import sha

PREFIX='Garnet/test2026/cuda/qwen38_5090/'
PARENT='native-cuda-phase-logged-bootstrap-attempt1'
NAMES=('native_cuda_phase_entry.py','test_native_cuda_phase_entry.py','serve_native_cuda_phase_entry_worker.bat',
       'probe_native_cuda_entry_embedded_cpu.py','probe_native_cuda_entry_bootstrap.py','collect_native_cuda_entry_profiled.py',
       'run_native_cuda_entry_profiled.py','verify_native_cuda_entry_profiled.py','prepare_native_cuda_entry_profiled.py',
       'test_native_cuda_entry_pipeline.py','evidence/native-cuda-entry-profiled-derivation.json')

def admit_parent(q,w,digest):
    assert q['status']=='native_cuda_logged_bootstrap_replayed' and q['archive_sha256']==digest
    assert q['profiled'] is False and q['performance_admitted'] is q['two_times_target_achieved'] is False
    assert type(w['exit_code']) is int and w['exit_code']==0 and w['cleanup_finished'] is w['library_unchanged'] is True
    assert w['source_sha256']==w['source_after_sha256'] and w['runtime_dependencies']==w['runtime_dependencies_after']
    assert q['library_sha256']==w['library_sha256']==w['runtime_dependencies']['garnet.dll']=='1a1d0229a1c0c24a8397a4dee7a614bcedf97b3bd3293e58d752ed7de30eb498'
    rows=q['control_replay']['requests'];assert len(rows)==1 and rows[0]['tokens_bound_to_retained_4k_control'] is True and rows[0]['mode']=='ordinary'

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
    failed=args.proofs/'native-cuda-phase-full4k-attempt1-evidence.zip'
    assert sha(failed)==q['linked_parent_archive_sha256']
    with zipfile.ZipFile(failed) as z:old=json.loads(z.read('results/native-cuda-phase-full4k-attempt1/workflow.json'))
    assert old['profiler']==old['profiler_after'] and old['cleanup_finished'] and old['exit_code']==1
    files=dict(w['source_after_sha256']);new={PREFIX+n:sha(m/n) for n in NAMES};assert not(set(files)&set(new));files.update(new)
    candidate=json.loads((m/'evidence/bf16-scores-tiled-native-api-build-preparation.json').read_bytes())['candidate_files']
    for n,h in files.items():
        if n.startswith('Garnet/'):
            relative=n.removeprefix('Garnet/');assert sha(repo/relative)==candidate.get(relative,h),('Baseline/candidate source drift',n)
    for n in NAMES:
        if n.endswith('.py'):ast.parse((m/n).read_bytes())
    modules=['test2026.cuda.qwen38_5090.'+n for n in ('test_native_cuda_phases','test_native_cuda_bootstrap_log','test_native_cuda_phase_entry','test_native_cuda_entry_pipeline')]
    assert all('Garnet/'+n.replace('.','/')+'.py' in files for n in modules)
    def ids(suite):
        for t in suite:
            if isinstance(t,unittest.TestSuite):yield from ids(t)
            else:yield t.id()
    loader=unittest.TestLoader();tests=sorted(ids(loader.loadTestsFromNames(modules)))
    assert not loader.errors and len(tests)==len(set(tests))==21
    from test2026.cuda.qwen38_5090.run_logged import run
    os.chdir(repo);code=run([sys.executable,'-m','unittest',*modules],args.work/'native-entry-profiled-cpu-ledger','correctness',
        '21CPU-only import/bootstrap/phase/raw-entry/parent and exact original-probe/owner derivation faults',[m/n for n in NAMES])
    assert code==0
    p={'status':'native_cuda_entry_profiled_prepared_gpu_unrun','files':files,'new_files':new,'runtime_dependencies':w['runtime_dependencies_after'],
       'parent_archive_sha256':digest,'parent_proof_sha256':sha(path),'profiler':old['profiler'],
       'test_modules':modules,'test_ids':tests,'expected_cpu_tests':21,
       'scope':'One exclusive actual unchanged B1 ordinary3968+128 retained text control under pinned Nsight2026.5 process-tree capture, normal inner IPC and early native import/exception log. Actual closed two-machine unprofiled bootstrap parent, source/runtime pins,21CPUguards and real embedded fake-import/fake-IO compatibility required. Actual model/vision construction, identical IDs,12unique HtoD markers, SM120 owned graph-node kernels/API/copy activity, raw SQLite/nsys reports and independent replay required for diagnostic success. Retain setup/native/profiler failures and original exit; source/runtime unchanged and owned cache retirement mandatory. Instrumented GPU activity is not HTTP speedup or serving-memory saving. Full text/image ordinary/MTP profiling, batched/quality/HTTP, wider prefill, strongest tuned references/repetitions/INT4/4B and overall2x remain open.'}
    target=m/'evidence/native-cuda-entry-profiled-preparation.json';raw=json.dumps(p,indent=2).encode()
    assert not target.exists() or target.read_bytes()==raw
    if not target.exists():target.write_bytes(raw)
    rows=[{'local':str(m/n),'target':PREFIX+n,'sha256':sha(m/n)} for n in NAMES]
    rows.append({'local':str(target),'target':PREFIX+'evidence/'+target.name,'sha256':sha(target)})
    (args.work/'native-entry-profiled-upload-plan.json').write_bytes(json.dumps({'uploads':rows},indent=2).encode())
    print(json.dumps({'status':p['status'],'source_pins':len(files),'new_targets':len(rows),'cpu_tests':21,'preparation_sha256':sha(target)}));return 0

if __name__=='__main__':raise SystemExit(main())
