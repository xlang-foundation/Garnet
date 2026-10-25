"""Own exclusive full4K correctness after the actual reference diagnosis closes."""
from pathlib import Path
import argparse,hashlib,json,os,re,shutil,sys,time,zipfile
if __package__:
    from .run_bf16_batch8_admission import retire
    from .run_mtp_prefill16_model import closed_native_admission
else:
    from run_bf16_batch8_admission import retire
    from run_mtp_prefill16_model import closed_native_admission
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()


def closed_characterization(root,name,digest):
    if re.fullmatch(r'sglang-output-characterization-attempt[1-9][0-9]*',name) is None or re.fullmatch('[0-9a-f]{64}',digest) is None:
        raise ValueError('Actual named closed characterization archive required')
    archive=root/'results'/(name+'-evidence.zip');assert sha(archive)==digest
    with zipfile.ZipFile(archive) as z:
        manifest=json.loads(z.read('evidence-manifest.json'));assert z.testzip() is None and len(z.namelist())==len(set(z.namelist()))==len(manifest['files'])+1
        for path,expected in manifest['files'].items():assert hashlib.sha256(z.read(path)).hexdigest()==expected
        w=json.loads(z.read('results/'+name+'/workflow.json'))
    assert w==json.loads((root/'results'/name/'workflow.json').read_bytes()) and type(w['exit_code']) is int and w['cleanup_finished'] is True and type(w['server_exit_code']) is int
    assert w['source_sha256']==w['source_after_sha256'] and w['runtime_dependencies']==w['runtime_dependencies_after']
    proof_path=root/'results'/(name+'-verified.json');proof=json.loads(proof_path.read_bytes());assert proof['archive_sha256']==digest
    assert proof['status']==('sglang_output_characterization_replayed' if w['exit_code']==0 else 'sglang_output_characterization_failure_preserved')
    return {'job':name,'archive_sha256':digest,'proof_sha256':sha(proof_path),'proof_status':proof['status'],'exit_code':w['exit_code'],'cleanup_finished':True}


def main():
    a=argparse.ArgumentParser(description=__doc__);a.add_argument('--closed-reference-job',required=True);a.add_argument('--closed-reference-archive-sha256',required=True);a.add_argument('--closed-native-archive-sha256',required=True);args=a.parse_args()
    root=Path('D:/CantorAI/5090-study-20261023');m=Path(__file__).resolve().parent;assert m==(root/'Garnet/test2026/cuda/qwen38_5090').resolve();sys.path.insert(0,str(m))
    from run_logged import run
    from run_bf16_low_batch_native_build import idle
    prep=m/'evidence/mtp-prefill16-full4k-run-preparation.json';p=json.loads(prep.read_bytes());assert p['status']=='mtp_prefill16_full4k_prepared_gpu_unrun'
    for name,digest in p['files'].items():assert sha(root/name)==digest,('Full4K input/source drift',name)
    for name,digest in p['runtime_dependencies'].items():assert sha(root/'build/bin'/name)==digest
    reference=closed_characterization(root,args.closed_reference_job,args.closed_reference_archive_sha256);native_closed=closed_native_admission(root,args.closed_native_archive_sha256)
    for name,row in p['parents'].items():
        proof=root/'results'/(name+'-verified.json');q=json.loads(proof.read_bytes());assert sha(proof)==row['proof_sha256'] and q['status']==row['status'] and sha(root/'results'/(name+'-evidence.zip'))==q['archive_sha256']==row['archive_sha256']
        closed=json.loads((root/'results'/name/'workflow.json').read_bytes());assert closed['exit_code']==0 and closed['cleanup_finished'] and closed['library_unchanged']
    receipt=root/'results/study-ssd-engine-cache/receipt.json';scratch=json.loads(receipt.read_bytes());cache=Path(scratch['cache_root'])
    assert scratch['status']=='ssd_engine_scratch_prepared' and cache.resolve().drive.upper()=='C:' and json.loads((cache/'study-owner.json').read_bytes())['study']==str(root) and shutil.disk_usage(cache).free>=96*1024**3
    name='mtp-prefill16-full4k-admission-attempt1';out=root/'results'/name;archive=root/'results'/(name+'-evidence.zip');assert not out.exists() and not archive.exists()
    idle();out.mkdir();os.chdir(root/'Garnet');code=1;before={f.name for f in cache.iterdir()};started=time.time();library=root/'build/bin/garnet.dll'
    common=[root/n for n in p['files']]+[prep,receipt,root/'results'/args.closed_reference_job/'workflow.json',root/'results'/(args.closed_reference_job+'-verified.json'),root/'results/bf16-batch8-model-admission-attempt1/workflow.json']+[root/'results'/(n+'-verified.json') for n in p['parents']]
    w={'status':'running','preparation_sha256':sha(prep),'source_sha256':p['files'],'runtime_dependencies':p['runtime_dependencies'],'library_sha256':sha(library),'parents':p['parents'],'closed_reference':reference,'closed_native_admission':native_closed,'cleanup_finished':False,'performance_admitted':False,'scope':p['scope']}
    def save():(out/'workflow.json').write_text(json.dumps(w,indent=2),encoding='utf8')
    save()
    try:
        code=run([sys.executable,'-m','unittest',*p['test_modules']],out/'cpu-ledger','correctness','Six full4K cases/32state raw replay fault controls plus unchanged target/MTP and closed-reference ownership',common)
        if code:raise RuntimeError('Full4K CPU admission controls failed')
        command=[sys.executable,str(m/'monitor_native_probe.py'),'--study',str(root),'--output',str(out/'resources'),'--','cmd.exe','/d','/c',str(m/'test_mtp_prefill16_full4k.bat'),str(out/'native'),str(cache)]
        code=run(command,out/'gpu-ledger','correctness','Actual six3968+128 ordinary/candidate target-state/raw-vector/ID/callback/reset cases and extra draft T16 memory, no timing',common)
        if code:raise RuntimeError('Full4K candidate correctness failed')
        report=json.loads((out/'native/report.json').read_bytes());assert report['status']=='mtp_prefill16_full4k_collected_needs_replay';w['status']='mtp_prefill16_full4k_collected_needs_replay'
    except BaseException as error:w.update(status='failed',error=repr(error));code=code or 1
    finally:
        w.update(exit_code=code,source_after_sha256={n:sha(root/n) for n in p['files']},runtime_dependencies_after={n:sha(root/'build/bin'/n) for n in p['runtime_dependencies']},library_unchanged=sha(library)==w['library_sha256']);save()
        idle();w['retired_private_engine_files']=retire(cache,root,before,started);w['cleanup_finished']=True;save()
        paths=[f for f in out.rglob('*') if f.is_file()]+common;raw={f.relative_to(root).as_posix():f.read_bytes() for f in paths}
        with zipfile.ZipFile(archive,'x',zipfile.ZIP_DEFLATED) as z:
            for n,data in raw.items():z.writestr(n,data)
            z.writestr('evidence-manifest.json',json.dumps({'files':{n:hashlib.sha256(data).hexdigest() for n,data in raw.items()},'scope':p['scope']},indent=2))
        print(json.dumps({'status':w['status'],'archive_sha256':sha(archive),'bytes':archive.stat().st_size,'files':len(raw)}),flush=True)
    return code


if __name__=='__main__':raise SystemExit(main())
