"""Full-word independent replay of the real FP8 M128 control/intervention."""
from pathlib import Path
import argparse,hashlib,json,re,struct,sys,zipfile
sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
from test2026.cuda.qwen38_5090.prefill128_fp8_projection import replay_projection,input_oracle
JOB='prefill128-fp8-projection-attempt1';OLD='actual-layer0-fp8-control-attempt1';FAILED='bf16-prefill128-model-admission-attempt2';METHODS='Garnet/test2026/cuda/qwen38_5090/'
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
        base='results/'+JOB+'/';fixture='results/actual-layer0-fp8-fixture/';doc=lambda name:json.loads(z.read(name));raw=z.read('results/prefill128-fp8-projection-preparation.json');p=json.loads(raw);w=doc(base+'workflow.json')
        assert sha(raw)==w['preparation_sha256'] and type(w['exit_code']) is int and w['cleanup_finished'] is True
        assert w['source_before']==w['source_after']==p['files'] and w['runtime_before']==w['runtime_after']==p['runtime'] and w['environment_before']==w['environment_after']==p['environment']
        assert w['gpu_model_executed'] is w['application_library_loaded'] is w['performance_admitted'] is False and w['parents']==p['parents']
        for name,h in p['methods'].items():assert sha(z.read(name))==h
        for name in ('prefill128_fp8_projection.py','verify_prefill128_fp8_projection.py'):
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
        extra_sources={}
        for name in ('src/backends/tensorrt/modelopt_cublaslt.cpp','src/backends/tensorrt/modelopt_cublaslt.h','src/cuda/modelopt_fp8.cu','src/cuda/modelopt_fp8.cuh'):
            key='Garnet/'+name;digest=sha(z.read(key))
            if key in p['files']:assert digest==p['files'][key]
            else:extra_sources[key]=digest
            for phase in ('cpu','build','control','shared'):
                events=[json.loads(line) for line in z.read(base+phase+'-ledger/commands.jsonl').splitlines()]
                assert len(events)==2 and events[0]['event']=='started' and events[1]['event']=='finished'
                matches=[h for path,h in events[0]['artifact_sha256'].items() if path.replace('\\','/').lower()=='d:/cantorai/5090-study-20261023/'+key.lower()]
                assert matches==[digest]
        assert set(extra_sources)=={'Garnet/src/cuda/modelopt_fp8.cu','Garnet/src/cuda/modelopt_fp8.cuh'}
        record=doc(METHODS+'evidence/prefill128-fp8-projection-derivation.json');source=z.read(METHODS+record['source']);assert sha(source)==record['source_sha256'];text=source.decode()
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
            assert b'Ran 4 tests' in z.read(base+'cpu-ledger/commands.log') and b'\nOK' in z.read(base+'cpu-ledger/commands.log')
            for mode in ('control','shared'):
                assert w[mode+'_exit_code']==0
                replayed[mode]=replay_projection(lambda name:z.read(fixture+name),lambda name:z.read(base+mode+'/'+name),mode);assert replayed[mode]==w[mode+'_replay']
                log=z.read(base+mode+'-ledger/commands.log').decode();assert 'CUBLASLT_VERSION 130400' in log
                loaded=re.findall(r'CUBLASLT_DLL ([^\r\n]+)',log);assert len(loaded)==1 and loaded[0].replace('\\','/').lower()=='d:/cantorai/5090-study-20261023/tools/cuda13/nvidia/cu13/bin/x86_64/cublaslt64_13.dll'
                assert any(name.replace('\\','/').lower()==loaded[0].replace('\\','/').lower() and h==p['runtime']['cublasLt64_13.dll'] for name,h in p['environment'].items())
                selected=re.findall(r'ALGORITHM status=(\d+) id_query=(\d+) id=(\d+) workspace=(\d+) opaque=([0-9a-f]+)',log)
                assert len(selected)==6 and all(row[:2]==('0','0') and int(row[3])==4<<20 for row in selected)
                assert all(selected[i]==selected[i+1] for i in (0,2,4));algorithms[mode]=[{'m':m,'id':int(selected[2*i][2]),'opaque':selected[2*i][4]} for i,m in enumerate((16,32,128))]
                if mode=='shared':
                    assert all(row==selected[0] for row in selected)
                    checks=re.findall(r'SHARED_ALGORITHM_CHECK status=(\d+) state=(\d+) workspace=(\d+)',log)
                    assert len(checks)==4 and all(a==b=='0' and int(c)<=4<<20 for a,b,c in checks) and 'M16_ALGORITHM_M128_CANDIDATE_PASSED' in log
            assert algorithms['control'][0]==algorithms['shared'][0]
        workspace_failure=None
        if not passed:
            assert w['exit_code']==2 and w['control_exit_code']==0 and w['shared_exit_code']==2 and w['status']=='failed'
            read_fixture=lambda name:z.read(fixture+name)
            replayed['control']=replay_projection(read_fixture,lambda name:z.read(base+'control/'+name),'control')
            assert replayed['control']==w['control_replay']
            source=read_fixture('input-m16-k6144.f32');expected=read_fixture('expected-m16-n5120.bf16');scale=struct.unpack('<f',read_fixture('input-scale.f32'))[0]
            shared_words=0
            for m in (16,32,128):
                cast,quantized=input_oracle(source*(m//16),scale)
                assert z.read(base+'shared/cast-m'+str(m)+'.bf16')==cast and z.read(base+'shared/quantized-m'+str(m)+'.e4m3')==quantized
                for repeat in (0,1):
                    name=base+'shared/actual-m'+str(m)+'-r'+str(repeat)+'.bf16'
                    if m==128:assert name not in manifest['files']
                    else:assert z.read(name)==expected*(m//16);shared_words+=m*5120
            log=z.read(base+'shared-ledger/commands.log').decode();checks=re.findall(r'SHARED_ALGORITHM_CHECK status=(\d+) state=(\d+) workspace=(\d+)',log)
            assert checks==[('0','0','1311839'),('0','0','1311839'),('0','0','5246879')]
            assert 5246879>4<<20 and 'ERROR Native helper matmul failed' in log
            shared=re.findall(r'ALGORITHM status=0 id_query=0 id=(\d+) workspace=4194304 opaque=([0-9a-f]+)',log)
            assert len(shared)==4 and all(row==shared[0] for row in shared) and shared[0][0]=='35'
            for phase in ('control','shared'):
                log=z.read(base+phase+'-ledger/commands.log').decode();assert 'CUBLASLT_VERSION 130400' in log and 'CUBLASLT_DLL D:/CantorAI/5090-study-20261023/tools/cuda13/nvidia/cu13/bin/x86_64/cublasLt64_13.dll' in log
            workspace_failure={'shared_m16_m32_output_words_exact':shared_words,'shared_m128_cast_and_quantized_input_exact':True,'shared_m128_matmul_executed':False,'algo_check_status':0,'algo_check_state':0,'required_workspace_bytes':5246879,'allowed_workspace_bytes':4194304,'default_m128_changed_words_each_repeat':[row['different_from_repeated_m16_output_words'] for row in replayed['control']['cases'] if row['m']==128],'cause_of_standalone_rejection':'Declared4MiB workspace budget rejected valid checked algorithm needing5246879bytes; not a hardware or library unsupported claim','full_model_cause_established':False}
        result={'workspace_failure':workspace_failure,'original_unified_inventory_missing_primitives':extra_sources,'all_raw_native_source_artifact_hashes_bound_across_all_four_processes':True,'external_bound_verifier_sha256':sha(Path(__file__).read_bytes()),'status':'prefill128_fp8_real_projection_replayed' if passed else 'prefill128_fp8_projection_failure_preserved','archive_sha256':sha(archive.read_bytes()),'files_verified':len(manifest['files']),'algorithm_observations':algorithms,'complete_output_word_replays':replayed,'total_output_words_replayed':sum(value['complete_bf16_output_words_replayed'] for value in replayed.values()),'source_runtime_environment_fixture_unchanged':True,'gpu_model_executed':False,'full_model_mismatch_cause_established':False,'model_or_quality_or_performance_qualified':False,'two_times_target_achieved':False,'replay_verifier_sha256':sha(Path(__file__).read_bytes()),'scope':p['scope']}
    finally:z.close();old.close()
    target=args.proofs/(JOB+'-verified.json');raw=json.dumps(result,indent=2).encode();assert not target.exists() or target.read_bytes()==raw
    if not target.exists():target.write_bytes(raw)
    print(json.dumps({'status':result['status'],'proof_sha256':sha(raw),'algorithm_ids':{name:[row['id'] for row in values] for name,values in algorithms.items()},'output_words_replayed':result['total_output_words_replayed']}));return 0 if passed else 1

if __name__=='__main__':raise SystemExit(main())
