"""Verify actual native build evidence; observation build is not numerical admission."""
from pathlib import Path
import argparse,hashlib,io,json,sys,zipfile
sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
from test2026.cuda.qwen38_5090.verify_gdn_chunk_stage_witness_bound import checked
JOB='b4-qk-m16-build-attempt1';P='Garnet/test2026/cuda/qwen38_5090/';sha=lambda b:hashlib.sha256(b).hexdigest()
def main():
    a=argparse.ArgumentParser();a.add_argument('--proofs',type=Path,required=True);args=a.parse_args();archive=args.proofs/(JOB+'-evidence.zip');z,manifest=checked(archive);base='results/'+JOB+'/'
    try:
        doc=lambda n:json.loads(z.read(n));raw=z.read(P+'evidence/b4-qk-m16-build-preparation.json');p=json.loads(raw);w=doc(base+'workflow.json')
        assert sha(raw)==w['preparation_sha256'] and w['cleanup_finished'] is w['baseline_source_restored'] is w['baseline_library_restored'] is True
        assert w['source_before']==w['source_after']==p['base_files'] and w['methods']==w['methods_after']==p['methods']
        assert w['runtime_before']==w['runtime_after']==p['runtime'] and w['environment_before']==w['environment_after']==p['environment'] and w['checkpoint_after']==p['checkpoints']
        assert w['cache_after_sha256']==p['cache_before_sha256']==sha(z.read(base+'CMakeCache-before.txt'))
        assert w['candidate_files']==p['candidate_files'] and w['candidate_base']==p['candidate_base'] and len(p['candidate_files'])==79
        assert w['gpu_model_executed'] is w['numerical_or_performance_admitted'] is False
        for n,h in p['methods'].items():assert sha(z.read(n))==h
        for n,h in p['checkpoints'].items():assert sha(z.read(n))==h
        derivation=(Path(__file__).parent/'evidence/b4-qk-m16-native-build-derivation.json').read_bytes();d=next(row for row in json.loads(derivation)['derivations'] if row['target']==Path(__file__).name);original=Path(__file__).with_name(d['source']);reader=original.read_bytes();assert sha(reader)==p['methods'][P+original.name]==d['source_sha256']
        for c in d['changes']:assert reader.count(c['old'].encode())==c['count'];reader=reader.replace(c['old'].encode(),c['new'].encode())
        assert reader==Path(__file__).read_bytes() and sha(reader)==d['target_sha256']==p['methods'][P+Path(__file__).name] and sha(z.read(p['bundle']))==p['bundle_sha256']
        with zipfile.ZipFile(io.BytesIO(z.read(p['bundle']))) as source:
            assert source.testzip() is None and len(source.namelist())==len(set(source.namelist()))==79 and {n:sha(source.read(n)) for n in source.namelist()}==p['candidate_files']
        with zipfile.ZipFile(io.BytesIO(z.read(base+'original-source.zip'))) as old:
            expected={n:h for n,h in p['candidate_base'].items() if h is not None};assert len(expected)==14 and old.testzip() is None and {n:sha(old.read(n)) for n in old.namelist()}==expected
        for job,row in p['parents'].items():
            proof=z.read('results/'+job+'-verified.json');q=json.loads(proof);assert sha(proof)==row['proof_sha256'] and proof==(args.proofs/(job+'-verified.json')).read_bytes()
            assert q['archive_sha256']==row['archive_sha256']==sha((args.proofs/(job+'-evidence.zip')).read_bytes())
        cp=doc('results/b4-qk-m16-cpu-attempt1-verified.json');assert cp['exit_code']==0 and cp['status']=='B4T4_QK_M16_actual_CPU_contracts_replayed' and cp['factory_fault_cases']==406 and cp['profile']['cases']==1248 and cp['plugin']['GPU_operator_calls']==0
        assert sha(z.read('results/b4-qk-m16-build-inputs.json'))==p['input_inventory_sha256']
        passed=w['exit_code']==0
        def command(name,expected_sources):
            events=[json.loads(line) for line in z.read(base+name+'-ledger/commands.jsonl').splitlines()];assert len(events)==2 and events[0]['event']=='started' and events[1]['event']=='finished' and events[1]['exit_code']==0
            assert events[0]['source']['owned_source_sha256']==events[1]['source_after']['owned_source_sha256']==expected_sources
        if passed:
            assert w['status']=='b4_qk_m16_native_built_CPU_qualified_model_unqualified'
            before={**p['base_files'],**p['methods']};installed={**before,**{'Garnet/'+n:h for n,h in p['candidate_files'].items()}};assert w['installed_source']==installed
            with zipfile.ZipFile(args.proofs/'full-chunk-stage-build-attempt2-evidence.zip') as linked:
                parent=json.loads(linked.read(P+'evidence/full-chunk-stage-source-derivation.json'))['parent_files']
            parent_sources={**before,**{'Garnet/'+n:h for n,h in parent.items()}};assert w['parent_installed_source']==parent_sources
            command('cpu-before',parent_sources);assert b'Ran 83 tests' in z.read(base+'cpu-before-ledger/commands.log') and b'Ran 86 tests' in z.read(base+'cpu-before-ledger/commands.log')
            for name in ('profile','o-profile','factory-build','factory-controls','wire','o-wire','init-build','init-controls','o-init-build','o-init-controls','native-build','selector-cpu','embedded-selector-cpu','tokenizer-cpu'):command(name,installed)
            assert doc(base+'profile/report.json')==dict(status='Q128_profile_CPU_contract_passed',cases=278,selected=16,rejected=262,GPU_or_model_executed=False)
            wire=doc(base+'wire/report.json');assert wire['status']=='fp8_q128_plugin_cpu_contract_passed' and wire['wire_bytes']==20 and wire['wire_policy']==0x51313238 and wire['bad_wires']==35 and wire['bad_constructors']==18 and wire['bad_descriptors']==23 and wire['bad_formats']==11 and wire['policy_rejections']==8 and wire['cross_wire_rejections']==9 and wire['workspace_bytes']==4849664 and wire['unexpected_gpu_calls']==0 and wire['runtime_version']==wire['header_version']==101601
            from test2026.cuda.qwen38_5090.run_q128_factory_cpu import check,MODES
            factory=doc(base+'factory-controls/report.json');assert factory['status']=='Q128_actual_factory_CPU_fault_contract_replayed' and factory['cases']==182 and factory['modes']==len(MODES)==86
            for name,h in factory['executables'].items():assert sha(z.read(base+'factory/'+name))==h
            for i,row in enumerate(factory['rows']):
                stdout=z.read(base+'factory-controls/'+str(i)+'-stdout.log');stderr=z.read(base+'factory-controls/'+str(i)+'-stderr.log');assert sha(stdout)==row['stdout_sha256'] and sha(stderr)==row['stderr_sha256'] and json.loads(stdout)==row['report'];check(row['report'],stderr,row['mode'],row['policy']=='1',row['exe']=='unsupported.exe')
            from test2026.cuda.qwen38_5090.q128_plugin_init_contract import cpu_contract
            init=doc(base+'init-controls/report.json');assert init['status']=='actual_q128_plugin_initialize_CPU_policy_contracts_replayed' and init['actual_plugin_initialize_calls']==630 and init['actual_factory_stub_calls']==490 and len(init['policies'])==10 and sha(z.read(base+'init/contract.exe'))==init['executable_sha256']
            for i,row in enumerate(init['policies']):
                stdout=z.read(base+'init-controls/policy-'+str(i)+'-stdout.log');stderr=z.read(base+'init-controls/policy-'+str(i)+'-stderr.log');assert sha(stdout)==row['stdout_sha256'] and sha(stderr)==row['stderr_sha256'] and json.loads(stdout)==row['report'];assert cpu_contract(row['report'],stderr,row['policy_value']=='1')==row['replay']
            assert doc(base+'o-profile/report.json')==dict(status='O128_profile_CPU_contract_passed',cases=278,selected=16,rejected=262,GPU_or_model_executed=False)
            owire=doc(base+'o-wire/report.json');assert owire['status']=='fp8_o128_plugin_cpu_contract_passed' and owire['wire_bytes']==20 and owire['wire_policy']==0x4f313238 and owire['bad_wires']==36 and owire['bad_constructors']==18 and owire['bad_descriptors']==23 and owire['bad_formats']==11 and owire['policy_rejections']==8 and owire['cross_wire_rejections']==11 and owire['workspace_bytes']==9175040 and owire['unexpected_gpu_calls']==0 and owire['runtime_version']==owire['header_version']==101601
            from test2026.cuda.qwen38_5090.o128_plugin_init_contract import cpu_contract as o_cpu_contract
            oinit=doc(base+'o-init-controls/report.json');assert oinit['status']=='actual_o128_plugin_initialize_CPU_policy_contracts_replayed' and oinit['actual_plugin_initialize_calls']==720 and oinit['actual_factory_stub_calls']==560 and len(oinit['policies'])==10 and sha(z.read(base+'o-init/contract.exe'))==oinit['executable_sha256']
            for i,row in enumerate(oinit['policies']):
                stdout=z.read(base+'o-init-controls/policy-'+str(i)+'-stdout.log');stderr=z.read(base+'o-init-controls/policy-'+str(i)+'-stderr.log');assert sha(stdout)==row['stdout_sha256'] and sha(stderr)==row['stderr_sha256'] and json.loads(stdout)==row['report'];assert o_cpu_contract(row['report'],stderr,row['policy_value']=='1')==row['replay']
            from test2026.cuda.qwen38_5090.run_z128_factory_trace_native_build import validate_cuda_objects
            assert w['cuda_objects_before']==p['cuda_objects_before'] and validate_cuda_objects(w['cuda_objects_before'],w['cuda_objects_built'])==w['changed_cuda_objects']
            assert sha(z.read(w['candidate_saved']))==w['candidate_library_sha256']!=p['runtime']['garnet.dll']
        else:assert w['status']=='failed' and w.get('error')
        result=dict(status='b4_qk_m16_native_CPU_build_replayed' if passed else 'b4_qk_m16_native_build_failure_replayed',archive_sha256=sha(archive.read_bytes()),archive_bytes=archive.stat().st_size,archive_members=len(z.namelist()),exit_code=w['exit_code'],error=w.get('error'),replay_reader_derivation_sha256=sha(derivation),original_reader_sha256=d['source_sha256'],corrected_reader_sha256=d['target_sha256'],inherited_linked_parent_metadata_lookup=True,new_separate_O_contracts_checked=True,original_all_Q_O_factory_wire_initialize_selector_tokenizer_controls_preserved=True,new_GDN_M16_CPU_contracts_parent_checked=True,new_V_M16_CPU_contracts_parent_checked=True,new_QK_M16_CPU_contracts_parent_checked=True,B4_GPU_or_model_admitted=False,candidate_files=79,candidate_library_sha256=w.get('candidate_library_sha256'),source_runtime_SDK_checkpoint_CMake_restored=True,current_CPU_controls=83 if passed else None,historical_controls=86 if passed else None,actual_header_CPU_cases=278 if passed else None,GPU_model_or_matmul_executed=False,model_or_quality_or_performance_qualified=False,two_times_target_achieved=False)
    finally:z.close()
    target=args.proofs/(JOB+'-verified.json');raw=json.dumps(result,indent=2).encode();assert not target.exists() or target.read_bytes()==raw;target.write_bytes(raw);print(json.dumps(result))
if __name__=='__main__':main()
