"""Independent archive replay; SDK/plugin construction is not model arithmetic qualification."""
from pathlib import Path
import argparse,hashlib,io,json,sys,zipfile
sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
JOB='z128-reconstructed-plugin-build-attempt2';METHODS='Garnet/test2026/cuda/qwen38_5090/';sha=lambda raw:hashlib.sha256(raw).hexdigest()
def main():
    a=argparse.ArgumentParser();a.add_argument('--proofs',type=Path,required=True);args=a.parse_args();archive=args.proofs/(JOB+'-evidence.zip');base='results/'+JOB+'/'
    with zipfile.ZipFile(archive) as z:
        doc=lambda name:json.loads(z.read(name));manifest=doc('evidence-manifest.json');assert z.testzip() is None and len(z.namelist())==len(set(z.namelist()))==len(manifest['files'])+1
        for name,h in manifest['files'].items():assert sha(z.read(name))==h
        raw=z.read(METHODS+'evidence/z128-reconstructed-plugin-build-preparation-attempt2.json');p=json.loads(raw);w=doc(base+'workflow.json')
        assert sha(raw)==w['preparation_sha256'] and w['cleanup_finished'] is w['baseline_source_restored'] is w['baseline_library_restored'] is True
        assert w['source_before']==w['source_after']==p['base_files'] and w['methods']==w['methods_after']==p['methods'] and w['runtime_before']==w['runtime_after']==p['runtime'] and w['environment_before']==w['environment_after']==p['environment']
        assert w['cache_after_sha256']==p['cache_before_sha256']==sha(z.read(base+'CMakeCache-before.txt')) and w['checkpoint_after']==p['checkpoints']
        assert w['candidate_files']==p['candidate_files'] and w['candidate_base']==p['candidate_base'] and len(p['candidate_files'])==43
        assert w['gpu_model_executed'] is w['numerical_or_performance_admitted'] is False
        for name,h in p['methods'].items():assert sha(z.read(name))==h
        assert sha(Path(__file__).read_bytes())==p['methods'][METHODS+Path(__file__).name]
        assert sha(z.read(p['proposal']))==p['proposal_sha256'] and sha(z.read(p['bundle']))==p['bundle_sha256']
        proposal=doc(p['proposal']);assert all(p[k]==v for k,v in proposal.items() if k!='status')
        with zipfile.ZipFile(io.BytesIO(z.read(p['bundle']))) as source:
            assert len(source.namelist())==len(set(source.namelist()))==43 and source.testzip() is None
            assert {n:sha(source.read(n)) for n in source.namelist()}==p['candidate_files']
        with zipfile.ZipFile(io.BytesIO(z.read(base+'original-source.zip'))) as original:
            expected={n:h for n,h in p['candidate_base'].items() if h is not None};assert len(expected)==14 and original.testzip() is None
            assert {n:sha(original.read(n)) for n in original.namelist()}==expected
        for job,row in p['parents'].items():
            raw=z.read('results/'+job+'-verified.json');q=json.loads(raw)
            assert sha(raw)==row['proof_sha256'] and raw==(args.proofs/(job+'-verified.json')).read_bytes() and q['archive_sha256']==row['archive_sha256']==sha((args.proofs/(job+'-evidence.zip')).read_bytes())
        passed=w['exit_code']==0
        if passed:
            assert w['status']=='reconstructed_Z128_plugin_native_built_CPU_qualified_model_unqualified'
            expected={**p['base_files'],**p['methods'],**{'Garnet/'+n:h for n,h in p['candidate_files'].items()}};assert expected==w['installed_source']
            for name in ('cpu-source','new-wire','new-init-build','new-profile','old-wire','new-init','native-build','selector-cpu','embedded-selector-cpu','tokenizer-cpu'):
                events=[json.loads(line) for line in z.read(base+name+'-ledger/commands.jsonl').splitlines()];assert len(events)==2 and events[0]['event']=='started' and events[1]['event']=='finished' and type(events[1]['exit_code']) is int and events[1]['exit_code']==0
                assert events[0]['source']['owned_source_sha256']==events[1]['source_after']['owned_source_sha256']==expected
            log=z.read(base+'cpu-source-ledger/commands.log');assert b'Ran 86 tests' in log and b'Ran 7 tests' in log
            wire=doc(base+'new-wire/report.json');checks={'status':'fp8_z128_reconstructed_plugin_cpu_contract_passed','wire_bytes':20,'wire_policy':0x52313238,'bad_wires':34,'bad_constructors':18,'bad_descriptors':23,'bad_formats':11,'policy_rejections':8,'cross_wire_rejections':7,'workspace_bytes':4849664,'unexpected_gpu_calls':0,'enqueue_tested':False,'model_qualified':False,'performance_admitted':False}
            for k,v in checks.items():assert type(wire[k]) is type(v) and wire[k]==v,k
            assert wire['runtime_version']==wire['header_version']==101601 and wire['loaded_tensorrt_dll'].lower()=='d:/cantorai/5090-study-20261023/tools/tensorrt/bin/nvinfer_10.dll'
            profile=doc(base+'new-profile/report.json');assert profile==dict(status='reconstructed_Z128_profile_CPU_contract_passed',cases=158,rejections=62,selected=1,GPU_or_model_executed=False)
            from test2026.cuda.qwen38_5090.fp8_z128_native_build_contract import z_wire_contract
            assert z_wire_contract(doc(base+'old-wire/report.json'))==w['old_wire_contract']
            from test2026.cuda.qwen38_5090.z128_reconstructed_plugin_init_contract import cpu_contract
            init=doc(base+'new-init/report.json');assert init['status']=='actual_reconstructed_plugin_initialize_CPU_policy_contracts_replayed' and init['actual_plugin_initialize_calls']==450 and init['actual_factory_stub_calls']==350 and init['unexpected_GPU_operator_calls']==0
            assert sha(z.read(base+'new-init-build/contract.exe'))==init['executable_sha256'] and len(init['policies'])==10
            for i,(policy,row) in enumerate(zip((None,'','0','true','01',' 1','1 ','-1','2','1'),init['policies'])):
                stdout=z.read(base+'new-init/policy-'+str(i)+'-stdout.log');stderr=z.read(base+'new-init/policy-'+str(i)+'-stderr.log');assert row['policy_value']==policy and row['exit_code']==0 and sha(stdout)==row['stdout_sha256'] and sha(stderr)==row['stderr_sha256'] and json.loads(stdout)==row['report'] and cpu_contract(row['report'],stderr,policy=='1')==row['replay']
            from test2026.cuda.qwen38_5090.run_z128_factory_trace_native_build import validate_cuda_objects
            assert w['cuda_objects_before']==p['cuda_objects_before'] and validate_cuda_objects(w['cuda_objects_before'],w['cuda_objects_built'])==w['changed_cuda_objects']
            assert sha(z.read(w['candidate_saved']))==w['candidate_library_sha256']!=p['runtime']['garnet.dll']
        else:
            assert w['status']=='failed' and w.get('error')
        result={'status':'reconstructed_Z128_plugin_native_CPU_build_replayed' if passed else 'reconstructed_Z128_plugin_build_failure_replayed','archive_sha256':sha(archive.read_bytes()),'archive_bytes':archive.stat().st_size,'archive_members':len(z.namelist()),'exit_code':w['exit_code'],'source_pins':len(p['base_files']),'method_pins':len(p['methods']),'candidate_files':43,'source_runtime_SDK_checkpoint_CMake_restored':True,'candidate_library_sha256':w.get('candidate_library_sha256'),'CPU_source_controls':7 if passed else None,'historical_controls_in_frozen_parent_view':86 if passed else None,'actual_CPU_initialize_calls':450 if passed else None,'actual_CPU_factory_stub_calls':350 if passed else None,'GPU_model_or_matmul_executed':False,'model_numerical_or_quality_or_performance_qualified':False,'two_times_target_achieved':False}
    target=args.proofs/(JOB+'-verified.json');target.write_bytes(json.dumps(result,indent=2).encode());print(json.dumps(result));return 0
if __name__=='__main__':raise SystemExit(main())
