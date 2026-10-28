"""Independent real-weight/cache provenance and SDK full-word control replay."""
from pathlib import Path
import argparse,hashlib,json,math,re,struct,sys,zipfile
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
from test2026.cuda.qwen38_5090.fa_value_sdk_control import CASES,K,N,PARENT,parent_fixture,observe,replay,parse_log,verify_derivations
JOB='fa-value-sdk-control-attempt1';P='Garnet/test2026/cuda/qwen38_5090/';F='results/actual-fa-value-m4-m16-fixture/';BASE='results/'+JOB+'/';sha=lambda raw:hashlib.sha256(raw).hexdigest()
def checked(path):
    z=zipfile.ZipFile(path);mf=json.loads(z.read('evidence-manifest.json'));assert z.testzip() is None and len(z.namelist())==len(set(z.namelist()))==len(mf['files'])+1
    for name,h in mf['files'].items():assert sha(z.read(name))==h
    return z,mf
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--proofs',type=Path,required=True);a=parser.parse_args();archive=a.proofs/(JOB+'-evidence.zip');z,mf=checked(archive)
    try:
        raw=z.read(P+'evidence/fa-value-sdk-preparation.json');p=json.loads(raw);w=json.loads(z.read(BASE+'workflow.json'));assert sha(raw)==w['preparation_sha256'] and w['cleanup_finished'] and type(w['exit_code']) is int and w['exit_code'] in (0,1,2)
        for before,after,key in [('source_before','source_after','files'),('runtime_before','runtime_after','runtime'),('environment_before','environment_after','environment'),('checkpoint_before','checkpoint_after','checkpoints'),('cuda_objects_before','cuda_objects_after','cuda_objects')]:assert w[before]==w[after]==p[key]
        assert w['unchanged_before_after'] and w['parents']==p['parents'] and w['gpu_model_executed'] is w['application_library_loaded'] is w['performance_admitted'] is False
        assert len(p['cuda_objects'])==34 and w['cache_before_sha256']==w['cache_after_sha256']==p['cache_before_sha256']
        for name,h in p['methods'].items():assert sha(z.read(name))==h
        for name in ('verify_fa_value_sdk_control.py','fa_value_sdk_control.py','fa_value_sdk_log.py','prefill128_fp8_projection.py','decode_boundary_analysis.py','decode_boundary_layout.py','b4_b1_exact_gate.py','b4_b1_short_m16_gate.py'):assert Path(__file__).with_name(name).read_bytes()==z.read(P+name) and sha(z.read(P+name))==p['files'][P+name]
        for name in p['compile_sources']:assert sha(z.read('Garnet/'+name))==p['files']['Garnet/'+name] and all(name not in p[mode+'_source_files'] for mode in ('reference','candidate'))
        for name in ('fa-value-sdk-derivation.json','fa-value-sdk-owner-derivation.json'):verify_derivations(json.loads(z.read(P+'evidence/'+name)),lambda n:z.read(P+n))
        for job,row in p['parents'].items():
            raw=z.read('results/'+job+'-verified.json');q=json.loads(raw);assert sha(raw)==row['proof_sha256'] and q['status']==row['status'] and q['archive_sha256']==row['archive_sha256'];assert raw==(a.proofs/(job+'-verified.json')).read_bytes() and sha((a.proofs/(job+'-evidence.zip')).read_bytes())==row['archive_sha256']
        fixture=None;observation=algorithms=strict=None;algorithm_error=observation_error=strict_error=None
        if F+'manifest.json' in mf['files']:
            raw=z.read(F+'manifest.json');fixture=json.loads(raw);assert sha(raw)==w['fixture_manifest_sha256'];assert w['fixture_before']==w['fixture_after']=={n:v['sha256'] for n,v in fixture['files'].items()};assert mf['external_weights']=={}
            for name,value in fixture['files'].items():assert value['archive'] is True and len(z.read(F+name))==value['bytes'] and sha(z.read(F+name))==value['sha256']
            q=json.loads((a.proofs/(PARENT+'-verified.json')).read_bytes());assert q['exit_code']==0 and q['diagnostic_passed'] and not q['original_failed_model_job_promoted'];assert fixture['parent_archive_sha256']==p['parents'][PARENT]['archive_sha256'] and fixture['parent_proof_sha256']==p['parents'][PARENT]['proof_sha256']
            parent,pmf=checked(a.proofs/(PARENT+'-evidence.zip'))
            try:
                base='results/'+PARENT+'/';reference=json.loads(parent.read(base+'reference/report.json'));candidate=json.loads(parent.read(base+'candidate/report.json'));data,cases=parent_fixture(reference,candidate,lambda n:parent.read(base+'reference/'+n),lambda n:parent.read(base+'candidate/'+n));assert fixture['cases']==cases
                for name,raw in data.items():assert z.read(F+name)==raw
                indexraw=parent.read('models/Qwen3.8-27B-NVFP4/model.safetensors.index.json');assert sha(indexraw)==p['checkpoints']['models/Qwen3.8-27B-NVFP4/model.safetensors.index.json'];index=json.loads(indexraw)
            finally:parent.close()
            assert fixture['status']=='actual_FA3_step5_FA11_step1_SDK_fixture_prepared' and len(fixture['checkpoint_tensors'])==6
            for label,layer,step in CASES:
                key='model.language_model.layers.'+str(layer)+'.self_attn.v_proj'
                for suffix,filename,shape,weight in (('.weight',label+'-weight.e4m3',[N,K],True),('.input_scale',label+'-input-scale.f32',[],False),('.weight_scale',label+'-weight-scale.f32',[],False)):
                    origin=fixture['checkpoint_tensors'][filename];assert origin['key']==key+suffix and origin['shard']==index['weight_map'][key+suffix]
                    headerraw=z.read(F+origin['shard']+'-header.json');assert sha(headerraw)==origin['header_sha256'];header=json.loads(headerraw);entry=header[key+suffix];assert {k:v for k,v in origin.items() if k not in ('key','shard','header_sha256','shard_bytes')}==entry
                    assert entry['shape']==shape and (entry['dtype'] in ('F8_E4M3','F8_E4M3FN') if weight else entry['dtype']=='F32');start,end=entry['data_offsets'];raw=z.read(F+filename);assert type(start) is type(end) is int and 0<=start<end and end-start==len(raw)==(N*K if weight else 4) and 8+len(headerraw)+end<=origin['shard_bytes']
                    if weight:assert not np.any((np.frombuffer(raw,dtype='u1')&127)==127)
                    else:value=struct.unpack('<f',raw)[0];assert math.isfinite(value) and value>0
            assert sum(fixture['files'][label+'-weight.e4m3']['bytes'] for label,_,_ in CASES)==10<<20
        stages={}
        for stage in ('cpu','fixture','build','sdk'):
            name=BASE+stage+'-ledger/commands.jsonl'
            if name not in mf['files']:continue
            events=[json.loads(line) for line in z.read(name).splitlines()];assert len(events)==2 and events[0]['event']=='started' and events[1]['event']=='finished' and type(events[1]['exit_code']) is int
            assert events[0]['source']['owned_source_sha256']==events[1]['source_after']['owned_source_sha256']==p['files'];stages[stage]=events[1]['exit_code']
        if 'cpu' in stages:assert stages['cpu']==0 and b'Ran 14 tests' in z.read(BASE+'cpu-ledger/commands.log') and b'\nOK' in z.read(BASE+'cpu-ledger/commands.log')
        if 'executable_sha256' in w:assert sha(z.read(BASE+'build/control.exe'))==w['executable_sha256']
        if 'sdk' in stages:
            assert stages['sdk']==w['sdk_exit_code'];raw=z.read(BASE+'sdk-ledger/commands.log');log=raw.decode();paths=re.findall(r'CUBLASLT_DLL ([^\r\n]+)',log)
            if stages['sdk']==0:
                assert paths==['D:/CantorAI/5090-study-20261023/tools/cuda13/nvidia/cu13/bin/x86_64/cublasLt64_13.dll'] and 'NATIVE_CORRECTNESS_RESOURCE_REPORT_SAVED' in log;assert json.loads(z.read(BASE+'resources/resource-monitor.json'))['child_exit_code']==0
                try:algorithms=parse_log(raw)
                except AssertionError as error:algorithm_error=repr(error);assert w['exit_code']!=0 and 'algorithm_replay' not in w
                if algorithms is not None:
                    assert algorithms==w['algorithm_replay']
                    try:observation=observe(lambda n:z.read(F+n),lambda n:z.read(BASE+'actual/'+n))
                    except AssertionError as error:observation_error=repr(error);assert w['exit_code']!=0 and 'observation' not in w
                    if observation is not None:
                        assert observation==w['observation']==json.loads(z.read(BASE+'observation.json'))
                        try:strict=replay(lambda n:z.read(F+n),lambda n:z.read(BASE+'actual/'+n))
                        except AssertionError as error:strict_error=repr(error);assert w['exit_code']!=0 and 'strict_replay' not in w
                        if strict is not None:assert strict==w['strict_replay']
        if w['exit_code']==0:assert stages==dict(cpu=0,fixture=0,build=0,sdk=0) and w['status']=='actual_FA_value_SDK_control_collected_needs_independent_replay' and strict is not None
        else:assert w['status']=='failed' and isinstance(w['error'],str)
        passed=w['exit_code']==0 and strict is not None
        result=dict(status='actual_FA_value_SDK_control_replayed' if passed else 'actual_FA_value_SDK_control_failure_replayed',exit_code=w['exit_code'],error=w.get('error'),archive_sha256=sha(archive.read_bytes()),archive_bytes=archive.stat().st_size,files_verified=len(mf['files']),stage_exit_codes=stages,fixture_cases=fixture['cases'] if fixture else None,observation=observation,algorithm_replay=algorithms,algorithm_error=algorithm_error,observation_error=observation_error,strict_error=strict_error,strict_replay_passed=passed,source_runtime_SDK_checkpoint_CMake_all34_production_objects_unchanged=True,component_value_projection_algorithm_effect_established_for_two_real_cases=passed,full_Garnet_inactive_V_projection_outputs_captured=False,internal_fused_normalization_directly_captured=False,full_model_or_quality_or_performance_qualified=False,two_times_target_achieved=False,replay_verifier_sha256=sha(Path(__file__).read_bytes()),scope=p['scope'])
    finally:z.close()
    target=a.proofs/(JOB+'-verified.json');raw=json.dumps(result,indent=2).encode();assert not target.exists() or target.read_bytes()==raw
    if not target.exists():target.write_bytes(raw)
    print(json.dumps(dict(status=result['status'],proof_sha256=sha(raw),strict_replay_passed=result['strict_replay_passed'],exit_code=result['exit_code'])));return 0
if __name__=='__main__':raise SystemExit(main())
