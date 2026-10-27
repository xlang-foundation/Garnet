"""Independent archive replay of actual new native CPU contracts or failure."""
from pathlib import Path
import argparse,hashlib,json,sys,zipfile
sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
from test2026.cuda.qwen38_5090.b4_v_m16_source import sources
from test2026.cuda.qwen38_5090.run_b4_v_m16_factory_cpu import check,MODES
JOB='b4-v-m16-cpu-attempt2';P='Garnet/test2026/cuda/qwen38_5090/';BASE='results/'+JOB+'/';sha=lambda b:hashlib.sha256(b).hexdigest()
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--proofs',type=Path,required=True);a=parser.parse_args();archive=a.proofs/(JOB+'-evidence.zip')
    with zipfile.ZipFile(archive) as z:
        mf=json.loads(z.read('evidence-manifest.json'));assert z.testzip() is None and len(z.namelist())==len(set(z.namelist()))==len(mf['files'])+1
        for n,h in mf['files'].items():assert sha(z.read(n))==h
        raw=z.read(P+'evidence/b4-v-m16-cpu-v2-preparation.json');p=json.loads(raw);w=json.loads(z.read(BASE+'workflow.json'));assert sha(raw)==w['preparation_sha256'] and w['cleanup_finished'] and type(w['exit_code']) is int
        for first,last,key in [('source_before','source_after','files'),('runtime_before','runtime_after','runtime'),('environment_before','environment_after','environment'),('checkpoint_before','checkpoint_after','checkpoints'),('cuda_objects_before','cuda_objects_after','cuda_objects')]:assert w[first]==w[last]==p[key]
        assert w['cache_before_sha256']==w['cache_after_sha256']==p['cache_before_sha256'] and w['unchanged_before_after'] and len(p['cuda_objects'])==34
        assert w['gpu_model_executed'] is w['production_source_or_DLL_installed'] is w['numerical_or_performance_admitted'] is False
        for n,h in {**p['methods'],**p['compile_inputs']}.items():assert sha(z.read(n))==h
        for n in ('b4_v_m16_source.py','run_b4_v_m16_factory_cpu.py','verify_b4_v_m16_cpu_v2.py'):assert sha(Path(__file__).with_name(n).read_bytes())==p['methods'][P+n]
        _,candidate,record=sources(lambda n:z.read(P+'native_b4_v_m16_candidate/'+Path(n).name));assert record['candidate_files']==p['candidate_files'] and len(candidate)==73
        import io
        bundle=z.read(p['bundle']);assert sha(bundle)==p['bundle_sha256']
        with zipfile.ZipFile(io.BytesIO(bundle)) as source:assert source.testzip() is None and len(source.namelist())==len(set(source.namelist()))==73 and all(source.read(n)==b for n,b in candidate.items())
        for job,row in p['parents'].items():
            proof=z.read('results/'+job+'-verified.json');q=json.loads(proof);assert sha(proof)==row['proof_sha256'] and q['archive_sha256']==row['archive_sha256'] and q['status']==row['status'] and proof==(a.proofs/(job+'-verified.json')).read_bytes() and sha((a.proofs/(job+'-evidence.zip')).read_bytes())==row['archive_sha256']
        stages={}
        for name in ('python','profile','factory','factory-controls','plugin'):
            path=BASE+name+'-ledger/commands.jsonl'
            if path not in mf['files']:continue
            events=[json.loads(line) for line in z.read(path).splitlines()];assert len(events)==2 and events[0]['event']=='started' and events[1]['event']=='finished' and type(events[1]['exit_code']) is int
            assert events[0]['source']['owned_source_sha256']==events[1]['source_after']['owned_source_sha256']==p['files'];stages[name]=events[1]['exit_code'];assert w[name+'_exit_code']==stages[name]
        profile=None;plugin=None;factory_count=0
        if w['exit_code']==0:
            assert stages==dict(python=0,profile=0,factory=0,**{'factory-controls':0},plugin=0) and w['status']=='B4T4_V_M16_CPU_contracts_collected_model_unqualified'
            assert b'Ran 12 tests' in z.read(BASE+'python-ledger/commands.log') and b'\nOK' in z.read(BASE+'python-ledger/commands.log')
            profile=json.loads(z.read(BASE+'profile/report.json'));assert profile==dict(status='B4T4_V_M16_profile_CPU_contract_passed',cases=1209,selected=16,rejected=1193,GPU_or_model_executed=False)
            report=json.loads(z.read(BASE+'factory-controls/report.json'));assert report['status']=='B4T4_V_M16_actual_factory_CPU_fault_contract_replayed' and report['cases']==len(report['rows'])==212 and report['modes']==len(MODES)==94 and report['GPU_model_or_SDK_matmul_executed'] is False
            for name,h in report['executables'].items():assert sha(z.read(BASE+'factory/'+name))==h
            baseline={}
            for i,row in enumerate(report['rows']):
                out=z.read(BASE+'factory-controls/'+str(i)+'-stdout.log');err=z.read(BASE+'factory-controls/'+str(i)+'-stderr.log');assert sha(out)==row['stdout_sha256'] and sha(err)==row['stderr_sha256'] and json.loads(out)==row['report']
                check(row['report'],err,row['mode'],row['geometry'],row['policy']=='1',row['exe']=='unsupported.exe');key=(row['exe'],row['geometry'],row['mode'])
                if key in baseline:assert baseline[key]==row['report']
                else:baseline[key]=row['report']
            factory_count=212;plugin=json.loads(z.read(BASE+'plugin/report.json'));assert plugin['status']=='B4T4_V_M16_plugin_wire_and_initialize_CPU_contract_passed' and {n:plugin[n] for n in ('wire_rejections','constructor_rejections','descriptor_rejections','format_rejections','cross_wire_rejections','initialize_calls','factory_calls','GPU_operator_calls')}==dict(wire_rejections=45,constructor_rejections=24,descriptor_rejections=46,format_rejections=9,cross_wire_rejections=7,initialize_calls=7,factory_calls=5,GPU_operator_calls=0)
            assert plugin['TRT_version']==101601 and plugin['TRT_DLL'].replace('\\','/').lower()=='d:/cantorai/5090-study-20261023/tools/tensorrt/bin/nvinfer_10.dll'
        else:assert w['status']=='failed' and isinstance(w['error'],str)
        result=dict(status='B4T4_V_M16_actual_CPU_contracts_replayed' if w['exit_code']==0 else 'B4T4_V_M16_CPU_contract_failure_replayed',exit_code=w['exit_code'],error=w.get('error'),archive_sha256=sha(archive.read_bytes()),files_verified=len(mf['files']),stage_exit_codes=stages,profile=profile,factory_fault_cases=factory_count,plugin=plugin,source_runtime_SDK_checkpoint_all34objects_CMakeCache_unchanged=True,production_source_or_DLL_installed=False,SDK_or_model_or_quality_or_performance_qualified=False,two_times_target_achieved=False,replay_verifier_sha256=sha(Path(__file__).read_bytes()),scope=p['scope'])
    target=a.proofs/(JOB+'-verified.json');raw=json.dumps(result,indent=2).encode();assert not target.exists() or target.read_bytes()==raw
    if not target.exists():target.write_bytes(raw)
    print(json.dumps(dict(status=result['status'],exit_code=result['exit_code'],proof_sha256=sha(raw),factory_fault_cases=factory_count)));return 0
if __name__=='__main__':raise SystemExit(main())
