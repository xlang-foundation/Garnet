"""Exclusive default/tiled whole-model trial; always return to baseline DLL."""
from pathlib import Path
import hashlib,json,os,shutil,sys,time,zipfile

ROOT=Path('D:/CantorAI/5090-study-20261023');M=ROOT/'Garnet/test2026/cuda/qwen38_5090';JOB='bf16-scores-tiled-model-admission-attempt1'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    assert Path(__file__).resolve().parent==M.resolve();sys.path.insert(0,str(M))
    from run_logged import run
    from run_bf16_low_batch_native_build import idle
    from run_bf16_batch8_admission import retire
    prep=M/'evidence/bf16-scores-tiled-model-preparation.json';p=json.loads(prep.read_bytes())
    for n,h in p['files'].items():assert sha(ROOT/n)==h,('Model source drift',n)
    for n,h in p['runtime'].items():assert sha(ROOT/'build/bin'/n)==h
    for n,h in p['environment'].items():assert sha(Path(n))==h
    build_proof=ROOT/'results/bf16-scores-tiled-native-build-attempt1-verified.json';b=json.loads(build_proof.read_bytes())
    assert sha(build_proof)==p['build_proof_sha256'] and b['status']=='native_tiled_scores_build_cpu_replayed' and b['candidate_library_sha256']==p['candidate_library_sha256']
    closed=json.loads((ROOT/'results/bf16-scores-tiled-native-build-attempt1/workflow.json').read_bytes());assert closed['exit_code']==0 and closed['cleanup_finished'] and closed['baseline_library_restored'] and closed['baseline_source_restored']
    candidate=ROOT/'build/bin/garnet-native-bf16-scores-tiled-attempt1.dll';baseline=ROOT/'build/bin/garnet-native-tokenizer-nfc-attempt1.dll';library=ROOT/'build/bin/garnet.dll'
    assert sha(candidate)==p['candidate_library_sha256'] and sha(baseline)==p['runtime']['garnet.dll']
    receipt=ROOT/'results/study-ssd-engine-cache/receipt.json';cache=Path(json.loads(receipt.read_bytes())['cache_root'])
    assert cache.resolve().drive.upper()=='C:' and json.loads((cache/'study-owner.json').read_bytes())['study']==str(ROOT) and shutil.disk_usage(cache).free>=96*1024**3
    out=ROOT/'results'/JOB;archive=ROOT/'results'/(JOB+'-evidence.zip');assert not out.exists() and not archive.exists()
    idle();out.mkdir();os.chdir(ROOT/'Garnet');before={f.name for f in cache.iterdir()};started=time.time();code=1
    w={'status':'running','source_sha256':p['files'],'runtime_dependencies':p['runtime'],'environment_before':p['environment'],'candidate_library_sha256':sha(candidate),
       'baseline_library_sha256':sha(library),'build_proof_sha256':sha(build_proof),'preparation_sha256':sha(prep),'baseline_archive_sha256':p['baseline_archive_sha256'],
       'cleanup_finished':False,'performance_admitted':False,'modes':{},'scope':p['scope']}
    def save():(out/'workflow.json').write_bytes(json.dumps(w,indent=2).encode())
    save();common=[prep,build_proof,receipt]+[ROOT/n for n in p['new_files']]
    try:
        code=run([sys.executable,'-m','unittest',*p['test_ids']],out/'cpu-ledger','correctness','Ten full4K replay and independent retained-baseline fault controls',common)
        if code:raise RuntimeError('CPU model replay gate failed')
        library.write_bytes(candidate.read_bytes());assert sha(library)==p['candidate_library_sha256']
        for mode in ('default','tiled'):
            command=[sys.executable,str(M/'monitor_native_probe.py'),'--study',str(ROOT),'--output',str(out/(mode+'-resources')),'--',
                     'cmd.exe','/d','/c',str(M/'test_bf16_scores_tiled_model.bat'),str(out/mode),str(cache),mode]
            code=run(command,out/(mode+'-ledger'),'correctness','Actual new-DLL '+mode+' B1 ordinary/MTP six3968+128text/image cases versus retained complete state/raw-vector oracle',common+[library])
            w['modes'][mode]={'exit_code':code};save()
            idle()
            if code:raise RuntimeError('Actual '+mode+' model/state gate failed; no timing admission')
            report=json.loads((out/mode/'report.json').read_bytes())
            assert report['local_replay']['status']=='tiled_scores_full4k_retained_baseline_exact' and report['tiled_scores_enabled']==(mode=='tiled')
        w['status']='native_tiled_scores_model_collected_needs_replay'
    except BaseException as error:w.update(status='failed',error=repr(error));code=code or 1
    finally:
        idle();library.write_bytes(baseline.read_bytes())
        w.update(exit_code=code,source_after_sha256={n:sha(ROOT/n) for n in p['files']},runtime_dependencies_after={n:sha(ROOT/'build/bin'/n) for n in p['runtime']},
                 environment_after={n:sha(Path(n)) for n in p['environment']},baseline_library_restored=sha(library)==p['runtime']['garnet.dll'])
        w['retired_private_engine_files']=retire(cache,ROOT,before,started);w['cleanup_finished']=True;save()
        paths=[f for f in out.rglob('*') if f.is_file()]+common
        names={f.relative_to(ROOT).as_posix():f for f in paths}
        with zipfile.ZipFile(archive,'x',zipfile.ZIP_DEFLATED) as z:
            for n,f in names.items():z.write(f,n)
            z.writestr('evidence-manifest.json',json.dumps({'files':{n:sha(f) for n,f in names.items()},'linked_parent_archive_sha256':p['baseline_archive_sha256'],
                                                       'linked_build_archive_sha256':b['archive_sha256'],'scope':p['scope']},indent=2))
        print(json.dumps({'status':w['status'],'archive_sha256':sha(archive),'bytes':archive.stat().st_size,'files':len(names)}),flush=True)
    return code

if __name__=='__main__':raise SystemExit(main())
