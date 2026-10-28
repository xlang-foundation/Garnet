"""Exclusive packed-only retry after preserved representation-control failure; reuse actual closed full companion, no timing admission."""
from pathlib import Path
import hashlib,json,os,shutil,sys,time,zipfile
ROOT=Path('D:/CantorAI/5090-study-20261023');M=ROOT/'Garnet/test2026/cuda/qwen38_5090';JOB='b4-packed-cache-characterization-attempt1'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def owned_common_inputs(p,prep,receipt):
    job=p['partial_parent']['job'];assert job in p['parents']
    return [prep,receipt]+[ROOT/n for n in p['new_files']]+[ROOT/'results'/(name+'-verified.json') for name in p['parents']]+[ROOT/'results'/(job+'-partial-verified.json')]

def main():
    assert Path(__file__).resolve().parent==M.resolve();sys.path.insert(0,str(M))
    from run_logged import run
    from run_bf16_low_batch_native_build import idle
    from run_bf16_batch8_admission import retire
    prep=M/'evidence/b4-packed-cache-characterization-preparation.json';p=json.loads(prep.read_bytes())
    for n,h in p['files'].items():assert sha(ROOT/n)==h,('B8 external B4 reference source drift',n)
    for n,h in p['runtime'].items():assert sha(ROOT/'build/bin'/n)==h
    for n,h in p['environment'].items():assert sha(Path(n))==h
    for name,row in p['parents'].items():
        proof=ROOT/'results'/(name+'-verified.json');q=json.loads(proof.read_bytes());assert sha(proof)==row['proof_sha256'] and q['status']==row['status'] and sha(ROOT/'results'/(name+'-evidence.zip'))==q['archive_sha256']==row['archive_sha256']
        w=json.loads((ROOT/'results'/name/'workflow.json').read_bytes());assert type(w['exit_code']) is int and w['exit_code']==row['exit_code'] and w['cleanup_finished'] and w['source_sha256']==w['source_after_sha256'] and w['runtime_dependencies']==w['runtime_dependencies_after']
    receipt=ROOT/'results/study-ssd-engine-cache/receipt.json';cache=Path(json.loads(receipt.read_bytes())['cache_root'])
    assert cache.resolve().drive.upper()=='C:' and json.loads((cache/'study-owner.json').read_bytes())['study']==str(ROOT) and shutil.disk_usage(cache).free>=96*1024**3
    out=ROOT/'results'/JOB;archive=ROOT/'results'/(JOB+'-evidence.zip');assert not out.exists() and not archive.exists()
    idle();out.mkdir();os.chdir(ROOT/'Garnet');before={f.name for f in cache.iterdir()};started=time.time();code=1
    w={'status':'running','source_sha256':p['files'],'runtime_dependencies':p['runtime'],'environment_before':p['environment'],'preparation_sha256':sha(prep),'parents':p['parents'],
       'library_sha256':sha(ROOT/'build/bin/garnet.dll'),'cleanup_finished':False,'performance_admitted':False,'scope':p['scope']}
    def save():(out/'workflow.json').write_bytes(json.dumps(w,indent=2).encode())
    save();common=owned_common_inputs(p,prep,receipt)
    import run_logged as ledger
    ledger.source_identity=lambda:{'scope':'explicit owned cache diagnosis inventory; unrelated untracked files excluded','owned_source_sha256':p['files']}
    try:
        code=run([sys.executable,'-m','unittest',*p['test_ids']],out/'cpu-ledger','correctness','Existing50controls plus3explicit packed empty-state and partial-proof archive dependency faults; original full-reference gate unchanged',common)
        if code:raise RuntimeError('CPU B8 external-reference gate failed')
        for mode in ('packed',):
            idle();command=[sys.executable,str(M/'monitor_native_probe.py'),'--study',str(ROOT),'--output',str(out/(mode+'-resources')),'--','cmd.exe','/d','/c',str(M/'test_b4_packed_cache_characterization.bat'),str(out/mode),str(cache),mode]
            code=run(command,out/(mode+'-gpu-ledger'),'correctness','Actual same-input four-row short case, full or packed BF16 cache,8output IDs and complete expanded state hashes/raw vectors; differences retained, no performance admission',common)
            if code:raise RuntimeError('Cache-mode diagnosis collection failed: '+mode)
            report=json.loads((out/mode/'report.json').read_bytes());assert report['status']=='b4_packed_cache_characterization_collected_needs_replay' and report['mode']==mode
            w['last_completed_mode']=mode;save()
        w['status']='b4_packed_cache_characterization_collected_needs_replay'
    except BaseException as error:w.update(status='failed',error=repr(error));code=code or 1
    finally:
        w.update(exit_code=code,source_after_sha256={n:sha(ROOT/n) for n in p['files']},runtime_dependencies_after={n:sha(ROOT/'build/bin'/n) for n in p['runtime']},
                 environment_after={n:sha(Path(n)) for n in p['environment']},library_unchanged=sha(ROOT/'build/bin/garnet.dll')==w['library_sha256']);save()
        idle();w['retired_private_engine_files']=retire(cache,ROOT,before,started);w['cleanup_finished']=True;save()
        paths=[f for f in out.rglob('*') if f.is_file()]+common;names={f.relative_to(ROOT).as_posix():f for f in paths}
        with zipfile.ZipFile(archive,'x',zipfile.ZIP_DEFLATED) as z:
            for n,f in names.items():z.write(f,n)
            z.writestr('evidence-manifest.json',json.dumps({'files':{n:sha(f) for n,f in names.items()},'linked_parent_archives':{n:v['archive_sha256'] for n,v in p['parents'].items()},'scope':p['scope']},indent=2))
        print(json.dumps({'status':w['status'],'archive_sha256':sha(archive),'bytes':archive.stat().st_size,'files':len(names)}),flush=True)
    return code
if __name__=='__main__':raise SystemExit(main())
