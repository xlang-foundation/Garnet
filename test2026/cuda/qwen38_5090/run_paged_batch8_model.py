"""Exclusive preserved-profile B8 input integration; no MTP or timing admission."""
from pathlib import Path
import hashlib,json,os,shutil,sys,time,zipfile
ROOT=Path('D:/CantorAI/5090-study-20261023');M=ROOT/'Garnet/test2026/cuda/qwen38_5090';JOB='paged-batch8-model-preservation-attempt1'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    assert Path(__file__).resolve().parent==M.resolve();sys.path.insert(0,str(M))
    from run_logged import run
    from run_bf16_low_batch_native_build import idle
    from run_bf16_batch8_admission import retire
    prep=M/'evidence/paged-batch8-model-preparation.json';p=json.loads(prep.read_bytes())
    for n,h in p['files'].items():assert sha(ROOT/n)==h,('Paged B8 source drift',n)
    for n,h in p['runtime'].items():assert sha(ROOT/'build/bin'/n)==h
    for n,h in p['environment'].items():assert sha(Path(n))==h
    for name,row in p['parents'].items():
        proof=ROOT/'results'/(name+'-verified.json');q=json.loads(proof.read_bytes());assert sha(proof)==row['proof_sha256'] and q['status']==row['status'] and sha(ROOT/'results'/(name+'-evidence.zip'))==q['archive_sha256']==row['archive_sha256']
        w=json.loads((ROOT/'results'/name/'workflow.json').read_bytes());assert w['exit_code']==0 and w['cleanup_finished'] and w['source_sha256']==w['source_after_sha256'] and w['runtime_dependencies']==w['runtime_dependencies_after']
    receipt=ROOT/'results/study-ssd-engine-cache/receipt.json';cache=Path(json.loads(receipt.read_bytes())['cache_root'])
    assert cache.resolve().drive.upper()=='C:' and json.loads((cache/'study-owner.json').read_bytes())['study']==str(ROOT) and shutil.disk_usage(cache).free>=96*1024**3
    out=ROOT/'results'/JOB;archive=ROOT/'results'/(JOB+'-evidence.zip');assert not out.exists() and not archive.exists()
    idle();out.mkdir();os.chdir(ROOT/'Garnet');before={f.name for f in cache.iterdir()};started=time.time();code=1
    w={'status':'running','source_sha256':p['files'],'runtime_dependencies':p['runtime'],'environment_before':p['environment'],'preparation_sha256':sha(prep),'parents':p['parents'],
       'library_sha256':sha(ROOT/'build/bin/garnet.dll'),'cleanup_finished':False,'performance_admitted':False,'scope':p['scope']}
    def save():(out/'workflow.json').write_bytes(json.dumps(w,indent=2).encode())
    save();common=[prep,receipt]+[ROOT/n for n in p['new_files']]+[ROOT/'results'/(name+'-verified.json') for name in p['parents']]
    try:
        code=run([sys.executable,'-m','unittest',*p['test_ids']],out/'cpu-ledger','correctness','26strict old/new B8 witness, independent retained-oracle and paged input fault controls',common)
        if code:raise RuntimeError('CPU preservation gate failed')
        command=[sys.executable,str(M/'monitor_native_probe.py'),'--study',str(ROOT),'--output',str(out/'resources'),'--','cmd.exe','/d','/c',str(M/'test_paged_batch8_model.bat'),str(out/'model'),str(cache)]
        code=run(command,out/'gpu-ledger','correctness','Actual nine B8 short/full3968+128 text/image/permutation/reset cases; compare retained29complete states and58raw vectors/IDs, no model arithmetic or timing change',common)
        if code:raise RuntimeError('Actual paged B8 model/state preservation failed')
        report=json.loads((out/'model/report.json').read_bytes());assert report['retained_baseline_replay']['status']=='paged_batch8_retained_model_preserved'
        w['status']='paged_batch8_model_collected_needs_replay'
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
