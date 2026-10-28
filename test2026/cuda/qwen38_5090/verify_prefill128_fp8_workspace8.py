"""Full-word independent replay of the real FP8 M128 control/intervention."""
from pathlib import Path
import argparse,hashlib,json,re,sys,zipfile
sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
from test2026.cuda.qwen38_5090.prefill128_fp8_projection import replay_projection
JOB='prefill128-fp8-workspace8-attempt1';OLD='actual-layer0-fp8-control-attempt1';FAILED='bf16-prefill128-model-admission-attempt2';METHODS='Garnet/test2026/cuda/qwen38_5090/'
sha=lambda raw:hashlib.sha256(raw).hexdigest()

def checked(path):
    z=zipfile.ZipFile(path);manifest=json.loads(z.read('evidence-manifest.json'))
    assert z.testzip() is None and len(z.namelist())==len(set(z.namelist()))==len(manifest['files'])+1
    for name,h in manifest['files'].items():assert sha(z.read(name))==h
    return z,manifest

def main():
    a=argparse.ArgumentParser(description=__doc__);a.add_argument('--proofs',type=Path,required=True);args=a.parse_args()
    archive=args.proofs/(JOB+'-evidence.zip');z,manifest=checked(archive);old,old_manifest=checked(args.proofs/(OLD+'-evidence.zip'))
    try:
        base='results/'+JOB+'/';fixture='results/actual-layer0-fp8-fixture/';doc=lambda name:json.loads(z.read(name));raw=z.read('results/prefill128-fp8-workspace8-preparation.json');p=json.loads(raw);w=doc(base+'workflow.json')
        assert sha(raw)==w['preparation_sha256'] and type(w['exit_code']) is int and w['cleanup_finished'] is True
        assert w['source_before']==w['source_after']==p['files'] and w['runtime_before']==w['runtime_after']==p['runtime'] and w['environment_before']==w['environment_after']==p['environment']
        assert w['gpu_model_executed'] is w['application_library_loaded'] is w['performance_admitted'] is False and w['parents']==p['parents']
        for name,h in p['methods'].items():assert sha(z.read(name))==h
        for name in ('prefill128_fp8_projection.py','verify_prefill128_fp8_workspace8.py'):
            assert sha(Path(__file__).with_name(name).read_bytes())==p['methods'][METHODS+name]
        for name,row in p['parents'].items():
            proof=z.read('results/'+name+'-verified.json');q=json.loads(proof)
            assert sha(proof)==row['proof_sha256'] and q['status']==row['status'] and q['archive_sha256']==row['archive_sha256']==manifest['linked_parent_archives'][name]
            assert proof==(args.proofs/(name+'-verified.json')).read_bytes() and sha((args.proofs/(name+'-evidence.zip')).read_bytes())==row['archive_sha256']
        assert sha(z.read('results/'+FAILED+'-analysis.json'))==p['failure_analysis_sha256']
        fraw=z.read(fixture+'manifest.json');f=json.loads(fraw)
        assert sha(fraw)==w['fixture_manifest_sha256']==p['fixture_manifest_sha256'] and fraw==old.read(fixture+'manifest.json') and f['files']==p['fixture_files']
        assert w['fixture_after']=={name:row['sha256'] for name,row in p['fixture_files'].items()} and manifest['external_weight']==p['fixture_files']['weight-n5120-k6144.e4m3']
        for name,row in f['files'].items():
            if name=='weight-n5120-k6144.e4m3':continue
            raw=z.read(fixture+name);assert sha(raw)==row['sha256'] and len(raw)==row['bytes'] and raw==old.read(fixture+name)
        for name in ('src/backends/tensorrt/modelopt_cublaslt.cpp','src/backends/tensorrt/modelopt_cublaslt.h','src/cuda/modelopt_fp8.cu','src/cuda/modelopt_fp8.cuh'):
            assert sha(z.read('Garnet/'+name))==p['files']['Garnet/'+name]
        record=doc(METHODS+'evidence/prefill128-fp8-workspace8-derivation.json')['files'][0];source=z.read(METHODS+record['source']);assert sha(source)==record['source_sha256'];text=source.decode()
        for step in record['replacements']:
            assert text.count(step['old'])==step['count'];text=text.replace(step['old'],step['new'])
        assert text.encode()==z.read(METHODS+record['target']) and sha(text.encode())==record['target_sha256']
        passed=w['exit_code']==0;replayed={};algorithms={}
        if passed:
            assert w['status']=='prefill128_fp8_projection_collected_needs_replay' and sha(z.read(base+'build/control.exe'))==w['executable_sha256']
            for name in ('cpu','build','control','shared'):
                events=[json.loads(line) for line in z.read(base+name+'-ledger/commands.jsonl').splitlines()]
                assert len(events)==2 and events[0]['event']=='started' and events[1]['event']=='finished' and type(events[1]['exit_code']) is int and events[1]['exit_code']==0
                assert events[0]['source']['owned_source_sha256']==events[1]['source_after']['owned_source_sha256']==p['files']
            assert b'Ran 6 tests' in z.read(base+'cpu-ledger/commands.log') and b'\nOK' in z.read(base+'cpu-ledger/commands.log')
            for mode in ('control','shared'):
                assert w[mode+'_exit_code']==0
                replayed[mode]=replay_projection(lambda name:z.read(fixture+name),lambda name:z.read(base+mode+'/'+name),mode);assert replayed[mode]==w[mode+'_replay']
                log=z.read(base+mode+'-ledger/commands.log').decode();assert 'CUBLASLT_VERSION 130400' in log
                loaded=re.findall(r'CUBLASLT_DLL ([^\r\n]+)',log);assert len(loaded)==1 and loaded[0].replace('\\','/').lower()=='d:/cantorai/5090-study-20261023/tools/cuda13/nvidia/cu13/bin/x86_64/cublaslt64_13.dll'
                assert any(name.replace('\\','/').lower()==loaded[0].replace('\\','/').lower() and h==p['runtime']['cublasLt64_13.dll'] for name,h in p['environment'].items())
                selected=re.findall(r'ALGORITHM status=(\d+) id_query=(\d+) id=(\d+) workspace=(\d+) opaque=([0-9a-f]+)',log)
                assert len(selected)==6 and all(row[:2]==('0','0') and int(row[3])==(8<<20 if mode=='shared' and i>=4 else 4<<20) for i,row in enumerate(selected))
                assert all(selected[i]==selected[i+1] for i in (0,2,4));algorithms[mode]=[{'m':m,'id':int(selected[2*i][2]),'opaque':selected[2*i][4]} for i,m in enumerate((16,32,128))]
                if mode=='shared':
                    assert all(row[:3]+row[4:]==selected[0][:3]+selected[0][4:] for row in selected)
                    checks=re.findall(r'SHARED_ALGORITHM_CHECK status=(\d+) state=(\d+) workspace=(\d+)',log)
                    assert len(checks)==4 and all(a==b=='0' and int(c)<=(4<<20 if i<2 else 8<<20) for i,(a,b,c) in enumerate(checks)) and 'M16_ALGORITHM_M128_CANDIDATE_PASSED' in log
            assert algorithms['control'][0]==algorithms['shared'][0]
        result={'status':'prefill128_fp8_real_projection_replayed' if passed else 'prefill128_fp8_projection_failure_preserved','archive_sha256':sha(archive.read_bytes()),'files_verified':len(manifest['files']),'algorithm_observations':algorithms,'complete_output_word_replays':replayed,'total_output_words_replayed':sum(value['complete_bf16_output_words_replayed'] for value in replayed.values()),'source_runtime_environment_fixture_unchanged':True,'gpu_model_executed':False,'full_model_mismatch_cause_established':False,'model_or_quality_or_performance_qualified':False,'two_times_target_achieved':False,'replay_verifier_sha256':sha(Path(__file__).read_bytes()),'scope':p['scope']}
    finally:z.close();old.close()
    target=args.proofs/(JOB+'-verified.json');raw=json.dumps(result,indent=2).encode();assert not target.exists() or target.read_bytes()==raw
    if not target.exists():target.write_bytes(raw)
    print(json.dumps({'status':result['status'],'proof_sha256':sha(raw),'algorithm_ids':{name:[row['id'] for row in values] for name,values in algorithms.items()},'output_words_replayed':result['total_output_words_replayed']}));return 0 if passed else 1

if __name__=='__main__':raise SystemExit(main())
