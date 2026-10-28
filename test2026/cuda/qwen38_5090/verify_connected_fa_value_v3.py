"""Independent connected-component baseline or failure replay; no model promotion."""
from pathlib import Path
import argparse,hashlib,io,json,struct,sys,zipfile
sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
from test2026.cuda.qwen38_5090.connected_fa_value_gate import observe,admit
from test2026.cuda.qwen38_5090.fa_value_sdk_control import CASES,PARENT,parent_fixture,verify_derivations
from test2026.cuda.qwen38_5090.decode_boundary_analysis import validate_observations
JOB='connected-fa-value-attempt3';P='Garnet/test2026/cuda/qwen38_5090/';BASE='results/'+JOB+'/';F='results/connected-fa-value-fixture/';sha=lambda raw:hashlib.sha256(raw).hexdigest()
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--proofs',type=Path,required=True);a=parser.parse_args();archive=a.proofs/(JOB+'-evidence.zip')
    with zipfile.ZipFile(archive) as z:
        mf=json.loads(z.read('evidence-manifest.json'));assert z.testzip() is None and len(z.namelist())==len(set(z.namelist()))==len(mf['files'])+1
        for name,h in mf['files'].items():assert sha(z.read(name))==h
        raw=z.read(P+'evidence/connected-fa-value-v3-preparation.json');p=json.loads(raw);w=json.loads(z.read(BASE+'workflow.json'));assert sha(raw)==w['preparation_sha256'] and type(w['exit_code']) is int and w['cleanup_finished'] is w['baseline_source_restored'] is w['baseline_library_restored'] is True
        assert w['source_before']==w['source_after']==p['files'] and w['runtime_before']==w['runtime_after']==p['runtime'] and w['environment_before']==w['environment_after']==p['environment'] and w['checkpoint_after']==p['checkpoints']
        assert w['cuda_objects_before']==w['cuda_objects_after']==p['cuda_objects'] and w['cache_before_sha256']==w['cache_after_sha256']==p['cache_before_sha256'] and w['build_artifacts_unchanged'] and len(p['cuda_objects'])==34
        assert w['performance_admitted'] is w['full4k_MTP_805_qualified'] is False
        for name,h in p['archive_files'].items():assert sha(z.read(name))==h
        for name in ('verify_connected_fa_value_v3.py','connected_fa_value_runtime.py','connected_fa_value_layout.py','connected_fa_value_gate.py','fa_value_sdk_control.py','fa_value_sdk_log.py','decode_boundary_analysis.py','decode_boundary_layout.py','b4_b1_short_m16_gate.py','b4_b1_exact_gate.py','prefill128_fp8_projection.py','decode_fp8_sdk_control.py'):assert Path(__file__).with_name(name).read_bytes()==z.read(P+name)
        verify_derivations(json.loads(z.read(P+'evidence/connected-fa-value-owner-derivation.json')),lambda n:z.read(P+n))
        verify_derivations(json.loads(z.read(P+'evidence/connected-fa-value-v2-derivation.json')),lambda n:z.read(P+n))
        verify_derivations(json.loads(z.read(P+'evidence/connected-fa-value-v3-derivation.json')),lambda n:z.read(P+n))
        for mode in ('reference','candidate'):
            assert sha(z.read(p[mode+'_bundle']))==p[mode+'_bundle_sha256']
            with zipfile.ZipFile(io.BytesIO(z.read(p[mode+'_bundle']))) as bundle:assert bundle.testzip() is None and len(bundle.namelist())==len(set(bundle.namelist()))==len(p[mode+'_source_files']) and {n:sha(bundle.read(n)) for n in bundle.namelist()}==p[mode+'_source_files']
        for job,row in p['parents'].items():
            raw=z.read('results/'+job+'-verified.json');q=json.loads(raw);assert sha(raw)==row['proof_sha256'] and q['status']==row['status'] and raw==(a.proofs/(job+'-verified.json')).read_bytes() and sha((a.proofs/(job+'-evidence.zip')).read_bytes())==q['archive_sha256']==row['archive_sha256']
        old_sdk=json.loads((a.proofs/'fa-value-sdk-control-attempt1-verified.json').read_bytes());assert old_sdk['exit_code']==1 and not old_sdk['strict_replay_passed'];parent_proof=json.loads((a.proofs/(PARENT+'-verified.json')).read_bytes());assert parent_proof['exit_code']==0 and parent_proof['diagnostic_passed']
        fixture=None
        if F+'manifest.json' in mf['files']:
            raw=z.read(F+'manifest.json');fixture=json.loads(raw);assert sha(raw)==w['fixture_manifest_sha256'];assert fixture['parent_archive_sha256']==p['parents'][PARENT]['archive_sha256'] and w['fixture_files']=={n:v['sha256'] for n,v in fixture['files'].items()}
            for name,value in fixture['files'].items():assert len(z.read(F+name))==value['bytes'] and sha(z.read(F+name))==value['sha256']
            with zipfile.ZipFile(a.proofs/(PARENT+'-evidence.zip')) as old:
                base='results/'+PARENT+'/';reference=json.loads(old.read(base+'reference/report.json'));candidate=json.loads(old.read(base+'candidate/report.json'));r=lambda n:old.read(base+'reference/'+n);c=lambda n:old.read(base+'candidate/'+n);data,cases=parent_fixture(reference,candidate,r,c);collected=validate_observations(reference,r);assert cases==fixture['cases'];inputs=json.loads(z.read(F+'actual-hidden-inputs.json'))
                for name,raw in data.items():assert z.read(F+name)==raw
                for label,layer,step in CASES:
                    raw=b''.join(collected[step,row]['boundaries'][layer-1] for row in range(4));assert raw==z.read(F+label+'-step0-hidden.f32') and struct.pack('<'+str(len(inputs[label]))+'f',*inputs[label])==raw
        def commands(name,expected):
            events=[json.loads(line) for line in z.read(BASE+name+'-ledger/commands.jsonl').splitlines()];assert len(events)==2 and events[0]['event']=='started' and events[1]['event']=='finished' and type(events[1]['exit_code']) is int;assert events[0]['source']['owned_source_sha256']==events[1]['source_after']['owned_source_sha256']==expected;return events[1]['exit_code']
        assert commands('cpu',p['files'])==0 and b'Ran 12 tests' in z.read(BASE+'cpu-ledger/commands.log') and b'\nOK' in z.read(BASE+'cpu-ledger/commands.log')
        if fixture is not None:assert commands('fixture',p['files'])==0
        reports={}
        for phase in w['phases']:
            mode=phase['mode'];assert mode not in reports and mode in ('reference','candidate') and phase['source_installed']==p[mode+'_installed_files'] and phase['library_sha256']==p[mode+'_library_sha256'] and commands(mode,p[mode+'_installed_files'])==phase['exit_code']
            if BASE+mode+'/report.json' in mf['files']:
                report=json.loads(z.read(BASE+mode+'/report.json'));assert report['library_sha256']==p[mode+'_library_sha256'] and report['mode']==mode and report['fixture_manifest_sha256']==w['fixture_manifest_sha256'];reports[mode]=report
        with zipfile.ZipFile(a.proofs/'fa-value-sdk-control-attempt1-evidence.zip') as sdk:
            for report in reports.values():
                for record in report['cases']:
                    label=record['label'];prefix='model.language_model.layers.'+str(record['layer']);metadata=record['weight_metadata'];assert set(metadata)=={prefix+suffix for suffix in ('.mlp.gate_up_proj.weight','.mlp.down_proj.weight','.self_attn.q_proj.weight','.self_attn.k_proj.weight','.self_attn.v_proj.weight','.self_attn.o_proj.weight')};info=metadata[prefix+'.self_attn.v_proj.weight'];assert info['logical_shape']==[1024,5120] and info['packed_bytes']==1024*5120 and info['scale_bytes']==0 and info['recipe']=='modelopt_fp8_w8a8'
                    for field,filename in (('input_scale','input-scale'),('weight_scale','weight-scale')):assert info[field]==struct.unpack('<f',sdk.read('results/actual-fa-value-m4-m16-fixture/'+label+'-'+filename+'.f32'))[0]
        observation=None;admission=None;gate_error=None
        if set(reports)=={'reference','candidate'} and all(q['status']=='connected_FA_value_components_collected' for q in reports.values()):
            r=lambda n:z.read(BASE+'reference/'+n);c=lambda n:z.read(BASE+'candidate/'+n);f=lambda n:z.read(F+n)
            try:observation=observe(reports['reference'],reports['candidate'],r,c,f)
            except AssertionError as error:gate_error=repr(error);assert w['exit_code']!=0 and 'observation' not in w
            if observation is not None:
                assert observation==w['observation']==json.loads(z.read(BASE+'analysis.json'))
                try:admission=admit(reports['reference'],reports['candidate'],r,c,f)
                except AssertionError as error:gate_error=repr(error);assert w['exit_code']!=0 and 'component_admission' not in w
                if admission is not None:assert admission==w['component_admission']
        if w['exit_code']==0:assert admission is not None and w['status']=='connected_FA_value_component_baselines_exact_unqualified_model_or_performance' and [q['mode'] for q in w['phases']]==['reference','candidate'] and all(q['exit_code']==0 for q in w['phases'])
        else:assert w['status']=='failed' and isinstance(w['error'],str)
        result=dict(status='connected_FA_value_component_baselines_replayed' if admission is not None else 'connected_FA_value_component_failure_replayed',exit_code=w['exit_code'],error=w.get('error'),archive_sha256=sha(archive.read_bytes()),archive_bytes=archive.stat().st_size,archive_members=len(z.namelist()),phase_statuses={k:q['status'] for k,q in reports.items()},phase_errors={k:q.get('error') for k,q in reports.items()},observation=observation,gate_error=gate_error,component_baselines_exact=admission is not None,source_runtime_SDK_checkpoint_CMake_all34objects_restored=True,original_failed_SDK_and_model_jobs_promoted=False,internal_fused_normalization_directly_captured=False,full_model_or_quality_or_performance_qualified=False,two_times_target_achieved=False,replay_verifier_sha256=sha(Path(__file__).read_bytes()),scope=p['scope'])
    target=a.proofs/(JOB+'-verified.json');raw=json.dumps(result,indent=2).encode();assert not target.exists() or target.read_bytes()==raw
    if not target.exists():target.write_bytes(raw)
    print(json.dumps(dict(status=result['status'],exit_code=result['exit_code'],proof_sha256=sha(raw),component_baselines_exact=result['component_baselines_exact'])));return 0
if __name__=='__main__':raise SystemExit(main())
