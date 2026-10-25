"""Exclusive baseline-DLL I/O qualification only after the live model trial closes."""
from pathlib import Path
import argparse,hashlib,json,os,sys,zipfile
ROOT=Path('D:/CantorAI/5090-study-20261023');M=ROOT/'Garnet/test2026/cuda/qwen38_5090';JOB='paged-prompt-native-component-attempt1'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    a=argparse.ArgumentParser(description=__doc__);a.add_argument('--closed-native-archive-sha256',required=True);args=a.parse_args()
    assert Path(__file__).resolve().parent==M.resolve();sys.path.insert(0,str(M))
    from run_logged import run
    from run_bf16_low_batch_native_build import idle
    prep=M/'evidence/paged-prompt-native-component-preparation.json';p=json.loads(prep.read_bytes())
    parent='bf16-scores-tiled-api-model-admission-attempt1';proof=ROOT/'results'/(parent+'-verified.json');q=json.loads(proof.read_bytes());closed=json.loads((ROOT/'results'/parent/'workflow.json').read_bytes())
    assert q['status'] in ('native_tiled_scores_model_replayed','native_tiled_scores_model_failure_preserved')
    assert sha(proof)==p['closed_native_proof_sha256'] and q['archive_sha256']==p['closed_native_archive_sha256']==args.closed_native_archive_sha256==sha(ROOT/'results'/(parent+'-evidence.zip'))
    assert type(closed['exit_code']) is int and closed['cleanup_finished'] and closed['baseline_library_restored'] and closed['source_sha256']==closed['source_after_sha256'] and closed['runtime_dependencies']==closed['runtime_dependencies_after']
    for n,h in p['files'].items():assert sha(ROOT/n)==h,('Paged component source drift',n)
    for n,h in p['runtime'].items():assert sha(ROOT/'build/bin'/n)==h
    for n,h in p['environment'].items():assert sha(Path(n))==h
    for n,h in p['retained_fixture_files'].items():assert sha(ROOT/n)==h
    out=ROOT/'results'/JOB;archive=ROOT/'results'/(JOB+'-evidence.zip');assert not out.exists() and not archive.exists()
    idle();out.mkdir();os.chdir(ROOT/'Garnet');code=1
    w={'status':'running','preparation_sha256':sha(prep),'source_sha256':p['files'],'runtime_dependencies':p['runtime'],'environment_before':p['environment'],
       'closed_native_proof_sha256':sha(proof),'closed_native_archive_sha256':q['archive_sha256'],'cleanup_finished':False,'performance_admitted':False,'scope':p['scope']}
    def save():(out/'workflow.json').write_bytes(json.dumps(w,indent=2).encode())
    save();common=[prep,proof]+[ROOT/n for n in p['new_files']]
    try:
        code=run([sys.executable,'-m','unittest',*p['test_ids']],out/'cpu-ledger','correctness','Twelve paged input and independent raw-byte corruption controls',common)
        if code:raise RuntimeError('CPU paged source or replay controls failed')
        code=run([str(ROOT/'build/bin/xlang3.exe'),str(M/'probe_paged_prompt_embedded_cpu.py'),str(ROOT),str(out/'embedded-cpu')],out/'embedded-cpu-ledger',
                 'correctness','Same twelve fake-I/O and raw-byte controls inside actual embedded Python; no Garnet or GPU calls',common)
        if code:raise RuntimeError('Embedded CPU paged source/replay compatibility')
        code=run([sys.executable,str(M/'host_embedding_fixture.py'),str(ROOT/'models/Qwen3.8-27B-NVFP4'),str(out/'fixture')],out/'fixture-cpu-ledger','correctness',
                 'Independent selected checkpoint BF16 byte rows; verify exact original retained fixture',common)
        if code:raise RuntimeError('Independent checkpoint fixture preparation')
        for name,h in p['retained_fixture_files'].items():assert sha(out/'fixture'/Path(name).name)==h
        command=[sys.executable,str(M/'monitor_native_probe.py'),'--study',str(ROOT),'--output',str(out/'resources'),'--','cmd.exe','/d','/c',str(M/'test_paged_prompt_component.bat'),str(out/'fixture'),str(out/'native')]
        code=run(command,out/'gpu-ledger','correctness','Actual native eight-case B1/2/4/8 and B8/3968 byte/layout/view lifetime matrix; no decoder or benchmark',common)
        if code:raise RuntimeError('Native paged input component failed')
        report=json.loads((out/'native/report.json').read_bytes());assert report['status']=='paged_prompt_native_bytes_collected_needs_replay'
        w['status']='paged_prompt_native_component_collected_needs_replay'
    except BaseException as error:w.update(status='failed',error=repr(error));code=code or 1
    finally:
        w.update(exit_code=code,source_after_sha256={n:sha(ROOT/n) for n in p['files']},runtime_dependencies_after={n:sha(ROOT/'build/bin'/n) for n in p['runtime']},
                 environment_after={n:sha(Path(n)) for n in p['environment']});save();idle();w['cleanup_finished']=True;save()
        paths=[f for f in out.rglob('*') if f.is_file()]+common;names={f.relative_to(ROOT).as_posix():f for f in paths}
        with zipfile.ZipFile(archive,'x',zipfile.ZIP_DEFLATED) as z:
            for n,f in names.items():z.write(f,n)
            z.writestr('evidence-manifest.json',json.dumps({'files':{n:sha(f) for n,f in names.items()},'linked_native_archive_sha256':q['archive_sha256'],'scope':p['scope']},indent=2))
        print(json.dumps({'status':w['status'],'archive_sha256':sha(archive),'bytes':archive.stat().st_size,'files':len(names)}),flush=True)
    return code
if __name__=='__main__':raise SystemExit(main())
