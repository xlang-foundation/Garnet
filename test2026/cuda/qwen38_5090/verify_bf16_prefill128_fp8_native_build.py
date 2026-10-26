"""Replay combined T128/M128 FP8 native build/CPU contracts; no model/timing admission."""
from pathlib import Path
import argparse,hashlib,io,json,sys,zipfile
sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
JOB='bf16-prefill128-fp8-native-build-attempt1';METHODS='Garnet/test2026/cuda/qwen38_5090/'
sha=lambda raw:hashlib.sha256(raw).hexdigest()

def wire_contract(report,expected_dll):
    values={'status':'prefill128_plugin_cpu_contract_passed','profiles':16,'roundtrips':16,'clones':16,'bad_wires':832,'bad_descriptors':496,'bad_formats':336,'cross_dialect_rejections':128,'legacy_roundtrips':96,'unexpected_operator_calls':0,'enqueue_tested':False,'model_qualified':False,'performance_admitted':False}
    for name,value in values.items():assert type(report[name]) is type(value) and report[name]==value,name
    assert type(report['runtime_version']) is int and report['runtime_version']==report['header_version']
    assert report['loaded_tensorrt_dll'].replace('\\','/').lower()==expected_dll.replace('\\','/').lower()
    return values

def main():
    a=argparse.ArgumentParser(description=__doc__);a.add_argument('--proofs',type=Path,required=True);args=a.parse_args()
    archive=args.proofs/(JOB+'-evidence.zip');base='results/'+JOB+'/'
    with zipfile.ZipFile(archive) as z:
        doc=lambda name:json.loads(z.read(name));manifest=doc('evidence-manifest.json')
        assert z.testzip() is None and len(z.namelist())==len(set(z.namelist()))==len(manifest['files'])+1
        for name,h in manifest['files'].items():assert sha(z.read(name))==h
        raw=z.read(METHODS+'evidence/bf16-prefill128-fp8-native-build-preparation.json');p=json.loads(raw);w=doc(base+'workflow.json')
        assert sha(raw)==w['preparation_sha256'] and type(w['exit_code']) is int
        assert w['cleanup_finished'] is w['baseline_source_restored'] is w['baseline_library_restored'] is True
        assert w['source_before']==w['source_after']==p['base_files'] and w['runtime_before']==w['runtime_after']==p['runtime']
        assert w['environment_before']==w['environment_after']==p['environment'] and w['methods']==w['methods_after']==p['methods']
        assert w['candidate_files']==p['candidate_files'] and w['candidate_base']==p['candidate_base'] and w['parents']==p['parents']
        assert len(p['candidate_files'])==22 and sha(z.read(base+'CMakeCache-before.txt'))==p['cache_before_sha256']
        assert sha(z.read(p['proposal']))==p['proposal_sha256']
        proposal=doc(p['proposal']);assert proposal['status']=='bf16_prefill128_fp8_native_build_proposed_unrun'
        assert all(p[name]==value for name,value in proposal.items() if name!='status')
        for name,h in p['methods'].items():assert sha(z.read(name))==h
        assert sha(Path(__file__).read_bytes())==p['methods'][METHODS+Path(__file__).name]
        assert sha(z.read(p['bundle']))==p['bundle_sha256']
        with zipfile.ZipFile(io.BytesIO(z.read(p['bundle']))) as source:
            assert source.testzip() is None and len(source.namelist())==len(set(source.namelist()))==22 and set(source.namelist())==set(p['candidate_files'])
            for name,h in p['candidate_files'].items():assert sha(source.read(name))==h
        with zipfile.ZipFile(io.BytesIO(z.read(base+'original-source.zip'))) as original:
            expected={name:h for name,h in p['candidate_base'].items() if h is not None}
            assert original.testzip() is None and len(original.namelist())==len(set(original.namelist()))==len(expected) and set(original.namelist())==set(expected)
            for name,h in expected.items():assert sha(original.read(name))==h==p['base_files']['Garnet/'+name]
        for job,row in p['parents'].items():
            proof=z.read('results/'+job+'-verified.json');q=json.loads(proof)
            assert sha(proof)==row['proof_sha256'] and q['status']==row['status'] and q['archive_sha256']==row['archive_sha256']
            assert proof==(args.proofs/(job+'-verified.json')).read_bytes() and sha((args.proofs/(job+'-evidence.zip')).read_bytes())==row['archive_sha256']
        passed=w['exit_code']==0;wire=None
        if passed:
            assert w['status']=='bf16_prefill128_fp8_native_built_cpu_qualified_model_unqualified'
            expected={**p['base_files'],**p['methods'],**{'Garnet/'+name:h for name,h in p['candidate_files'].items()}}
            assert w['installed_source']==expected and w['gpu_model_executed'] is w['performance_admitted'] is False
            for name in ('cpu','build','wire-cpu','fp8-wire-cpu','selector-cpu','embedded-selector-cpu','tokenizer-cpu'):
                events=[json.loads(line) for line in z.read(base+name+'-ledger/commands.jsonl').splitlines()]
                assert len(events)==2 and events[0]['event']=='started' and events[1]['event']=='finished' and type(events[1]['exit_code']) is int and events[1]['exit_code']==0
                before=events[0]['source']['owned_source_sha256'];after=events[1]['source_after']['owned_source_sha256']
                assert before==after==(dict(p['base_files'],**p['methods']) if name=='cpu' else expected)
            log=z.read(base+'cpu-ledger/commands.log');assert ('Ran '+str(p['expected_cpu_test_count'])+' tests').encode() in log and b'\nOK' in log
            assert sha(z.read(w['candidate_saved']))==w['candidate_library_sha256']!=p['runtime']['garnet.dll']
            before=w['cuda_objects_before'];after=w['cuda_objects_built'];assert before==p['cuda_objects_before'] and set(before)<=set(after)
            added=sorted(set(after)-set(before));changed=sorted(name for name in before if after[name]!=before[name])
            from test2026.cuda.qwen38_5090.run_bf16_prefill128_fp8_native_build import validate_cuda_objects
            assert added==w['added_cuda_objects']==[] and changed==w['changed_cuda_objects']==validate_cuda_objects(before,after)
            dll='D:/CantorAI/5090-study-20261023/tools/TensorRT/bin/nvinfer_10.dll'
            wire=wire_contract(doc(base+'wire-cpu/report.json'),dll)
            from test2026.cuda.qwen38_5090.fp8_m128_native_control import wire_contract as fp8_wire_contract
            assert fp8_wire_contract(doc(base+'fp8-wire-cpu/report.json'))==w['fp8_wire_contract']
            assert dll.replace('/','\\') in p['environment']
            for name,status in (('selector-cpu','native_tiled_selector_cpu_exact'),('embedded-selector-cpu','embedded_native_tiled_selector_exact')):
                report=doc(base+name+'/report.json');assert report['status']==status and report['library_sha256']==w['candidate_library_sha256'] and report['gpu_executed'] is False
                values=(None,'','0','1','true','01',' 1','1 ','2','-1');assert len(report['cases'])==len(values)
                for i,(row,value) in enumerate(zip(report['cases'],values)):
                    actual=doc(base+name+'/'+str(i)+'-stdout.txt');assert row['actual']==actual and row['environment_value']==value and row['exit_code']==0
                    assert actual['first']==actual['after_environment_flip']==(value=='1') and actual['gpu_executed'] is False
            report=doc(base+'tokenizer-cpu/report.json');assert report['status']=='rebuilt_native_tokenizer_cpu_exact' and report['library_sha256']==w['candidate_library_sha256'] and report['cases']==955 and report['mismatches']==[]
            rows=z.read(base+'tokenizer-cpu/cases.jsonl');assert sha(rows)==report['raw_cases_sha256'];values=[json.loads(line) for line in rows.splitlines()]
            assert len(values)==955 and all(row['exact'] is True and row['code']==0 for row in values)
        result={'status':'bf16_prefill128_fp8_native_build_cpu_replayed' if passed else 'bf16_prefill128_fp8_native_build_failure_preserved','archive_sha256':sha(archive.read_bytes()),'files_verified':len(manifest['files']),'candidate_library_sha256':w.get('candidate_library_sha256'),'baseline_library_sha256':p['runtime']['garnet.dll'],'source_and_runtime_restored':True,'plugin_wire_cpu_contract':wire,'gpu_model_executed':False,'full_forward_or_model_qualified':False,'performance_admitted':False,'two_times_target_achieved':False,'verifier_sha256':sha(Path(__file__).read_bytes()),'scope':p['scope']}
    target=args.proofs/(JOB+'-verified.json');raw=json.dumps(result,indent=2).encode();assert not target.exists() or target.read_bytes()==raw
    if not target.exists():target.write_bytes(raw)
    print(json.dumps({'status':result['status'],'proof_sha256':sha(raw),'candidate_library_sha256':result['candidate_library_sha256']}));return 0 if passed else 1

if __name__=='__main__':raise SystemExit(main())
