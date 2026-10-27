"""Independent full-array replay of SDK intervention or its preserved failure."""
from pathlib import Path
import argparse,hashlib,json,re,struct,sys,zipfile
sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
from test2026.cuda.qwen38_5090.decode_fp8_sdk_control import observe,replay,parse_log,bf16_output,PROJECTIONS,K
JOB='decode-fp8-sdk-control-attempt1';P='Garnet/test2026/cuda/qwen38_5090/';F='results/actual-decode-fp8-m4-m16-fixture/';BASE='results/'+JOB+'/'
sha=lambda b:hashlib.sha256(b).hexdigest()

def checked(path):
    z=zipfile.ZipFile(path);mf=json.loads(z.read('evidence-manifest.json'));assert z.testzip() is None and len(z.namelist())==len(set(z.namelist()))==len(mf['files'])+1
    for n,h in mf['files'].items():assert sha(z.read(n))==h
    return z,mf

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--proofs',type=Path,required=True);args=parser.parse_args();archive=args.proofs/(JOB+'-evidence.zip');z,mf=checked(archive)
    try:
        praw=z.read(P+'evidence/decode-fp8-sdk-preparation.json');p=json.loads(praw);w=json.loads(z.read(BASE+'workflow.json'));assert sha(praw)==w['preparation_sha256'] and w['cleanup_finished'] and type(w['exit_code']) is int and w['exit_code'] in (0,1,2)
        for a,b,key in [('source_before','source_after','files'),('runtime_before','runtime_after','runtime'),('environment_before','environment_after','environment'),('checkpoint_before','checkpoint_after','checkpoints'),('cuda_objects_before','cuda_objects_after','cuda_objects')]:assert w[a]==w[b]==p[key]
        assert w['unchanged_before_after'] and w['parents']==p['parents'] and w['gpu_model_executed'] is w['application_library_loaded'] is w['performance_admitted'] is False
        assert len(p['cuda_objects'])==34
        for n,h in p['methods'].items():assert sha(z.read(n))==h
        for name in ('verify_decode_fp8_sdk_control.py','decode_fp8_sdk_control.py','prefill128_fp8_projection.py'):
            assert sha(Path(__file__).with_name(name).read_bytes())==p['files'][P+name]
        for name in p['compile_sources']:assert sha(z.read('Garnet/'+name))==p['files']['Garnet/'+name] and all(name not in p[mode+'_source_files'] for mode in ('reference','candidate'))
        for job,row in p['parents'].items():
            raw=z.read('results/'+job+'-verified.json');q=json.loads(raw);assert sha(raw)==row['proof_sha256'] and q['status']==row['status'] and q['archive_sha256']==row['archive_sha256']
            assert raw==(args.proofs/(job+'-verified.json')).read_bytes() and sha((args.proofs/(job+'-evidence.zip')).read_bytes())==row['archive_sha256']
        fixture=None;observation=None;algorithms=None;strict=None
        if F+'manifest.json' in mf['files']:
            fraw=z.read(F+'manifest.json');fixture=json.loads(fraw);assert sha(fraw)==w['fixture_manifest_sha256']
            assert w['fixture_before']==w['fixture_after']=={n:v['sha256'] for n,v in fixture['files'].items()}
            assert mf['external_weights']=={n:v for n,v in fixture['files'].items() if not v['archive']} and len(mf['external_weights'])==2
            for n,v in fixture['files'].items():
                if v['archive']:assert len(z.read(F+n))==v['bytes'] and sha(z.read(F+n))==v['sha256']
                else:assert F+n not in mf['files'] and n in ('qkv-weight.e4m3','z-weight.e4m3')
            parent='decode-projection-diagnostic-attempt3';assert fixture['parent_proof_sha256']==p['parents'][parent]['proof_sha256'] and fixture['parent_archive_sha256']==p['parents'][parent]['archive_sha256'] and fixture['token_inputs']==p['token_inputs']
            old,omf=checked(args.proofs/(parent+'-evidence.zip'))
            try:
                b='results/'+parent+'/';ref=json.loads(old.read(b+'reference/report.json'));candidate=json.loads(old.read(b+'candidate/report.json'));assert ref['token_inputs']==candidate['token_inputs']==p['token_inputs']
                for step in range(5):
                    rows=ref['cases'][4*step:4*step+4];c=candidate['cases'][step];raw=b''.join(old.read(b+'reference/'+r['repeats'][0]['normalized']['file']) for r in rows)
                    assert len(raw)==16*K*4 and raw==old.read(b+'candidate/'+c['repeats'][0]['normalized']['file'])==z.read(F+'step'+str(step)+'-normalized.f32') and raw!=raw[:4*K*4]*4
                    for label,n in PROJECTIONS.items():
                        small=b''.join(old.read(b+'reference/'+r['repeats'][0][label]['file']) for r in rows);large=old.read(b+'candidate/'+c['repeats'][0][label]['file']);stem=label+'-step'+str(step)
                        assert bf16_output(small,16*n)==z.read(F+stem+'-garnet-m4.bf16') and bf16_output(large,16*n)==z.read(F+stem+'-garnet-m16.bf16')
                for label,n in PROJECTIONS.items():
                    key='model.language_model.layers.0.linear_attn.in_proj_'+label
                    for suffix,field in (('input','input_scale'),('weight','weight_scale')):assert struct.unpack('<f',z.read(F+label+'-'+suffix+'-scale.f32'))[0]==ref['weight_metadata'][key+'.weight'][field]
                    origin=fixture['checkpoint_tensors'][label+'-weight.e4m3'];assert origin['key']==key+'.weight' and origin['shape']==[n,K] and origin['dtype'] in ('F8_E4M3','F8_E4M3FN') and fixture['files'][label+'-weight.e4m3']['bytes']==n*K
            finally:old.close()
        stages={}
        for stage in ('cpu','fixture','build','sdk'):
            name=BASE+stage+'-ledger/commands.jsonl'
            if name not in mf['files']:continue
            events=[json.loads(line) for line in z.read(name).splitlines()];assert len(events)==2 and events[0]['event']=='started' and events[1]['event']=='finished' and type(events[1]['exit_code']) is int
            assert events[0]['source']['owned_source_sha256']==events[1]['source_after']['owned_source_sha256']==p['files'];stages[stage]=events[1]['exit_code']
        if 'cpu' in stages:assert stages['cpu']==0 and b'Ran 4 tests' in z.read(BASE+'cpu-ledger/commands.log') and b'\nOK' in z.read(BASE+'cpu-ledger/commands.log')
        if 'executable_sha256' in w:assert sha(z.read(BASE+'build/control.exe'))==w['executable_sha256']
        if 'sdk' in stages:
            assert stages['sdk']==w['sdk_exit_code'];raw=z.read(BASE+'sdk-ledger/commands.log');log=raw.decode();paths=re.findall(r'CUBLASLT_DLL ([^\r\n]+)',log)
            if stages['sdk']==0:
                assert paths==['D:/CantorAI/5090-study-20261023/tools/cuda13/nvidia/cu13/bin/x86_64/cublasLt64_13.dll'];assert 'NATIVE_CORRECTNESS_RESOURCE_REPORT_SAVED' in log
                resource=json.loads(z.read(BASE+'resources/resource-monitor.json'));assert resource['child_exit_code']==0
                algorithms=parse_log(raw);assert algorithms==w['algorithm_replay'];observation=observe(lambda n:z.read(F+n),lambda n:z.read(BASE+'actual/'+n));assert observation==w['observation']==json.loads(z.read(BASE+'observation.json'))
                if w['exit_code']==0:strict=replay(lambda n:z.read(F+n),lambda n:z.read(BASE+'actual/'+n));assert strict==w['strict_replay']
        if w['exit_code']==0:assert stages==dict(cpu=0,fixture=0,build=0,sdk=0) and w['status']=='actual_decode_fp8_SDK_control_collected_needs_independent_replay' and strict is not None
        else:assert w['status']=='failed' and isinstance(w['error'],str)
        result=dict(status='actual_decode_fp8_SDK_control_replayed' if w['exit_code']==0 else 'actual_decode_fp8_SDK_control_failure_replayed',exit_code=w['exit_code'],error=w.get('error'),archive_sha256=sha(archive.read_bytes()),files_verified=len(mf['files']),stage_exit_codes=stages,observation=observation,algorithm_replay=algorithms,strict_replay_passed=strict is not None,source_runtime_SDK_checkpoint_all34_production_objects_unchanged=True,component_projection_algorithm_effect_established=strict is not None,internal_fused_normalization_directly_captured=False,full_model_or_quality_or_performance_qualified=False,two_times_target_achieved=False,replay_verifier_sha256=sha(Path(__file__).read_bytes()),scope=p['scope'])
    finally:z.close()
    target=args.proofs/(JOB+'-verified.json');raw=json.dumps(result,indent=2).encode();assert not target.exists() or target.read_bytes()==raw
    if not target.exists():target.write_bytes(raw)
    print(json.dumps(dict(status=result['status'],proof_sha256=sha(raw),strict_replay_passed=result['strict_replay_passed'],exit_code=result['exit_code'])));return 0

if __name__=='__main__':raise SystemExit(main())
