from pathlib import Path
import argparse,hashlib,json,sys
sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
from test2026.cuda.qwen38_5090.verify_gdn_chunk_stage_witness_bound import checked
JOB='qg128-sdk-control-attempt2';P='Garnet/test2026/cuda/qwen38_5090/';sha=lambda b:hashlib.sha256(b).hexdigest()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--proofs',type=Path,required=True);a=ap.parse_args();archive=a.proofs/(JOB+'-evidence.zip');z,manifest=checked(archive);base='results/'+JOB+'/'
    try:
        p=json.loads(z.read(P+'evidence/qg128-sdk-control-preparation-attempt2.json'));w=json.loads(z.read(base+'workflow.json'));assert w['cleanup_finished'] and w['no_model_weights_or_matmul'] and w['source_before']==w['source_after'] and w['runtime_before']==w['runtime_after'] and w['SDK_before']==w['SDK_after'] and w['checkpoint_before']==w['checkpoint_after'];assert sha(z.read(P+'evidence/qg128-sdk-control-preparation-attempt2.json'))==w['preparation_sha256']
        for n,h in p['files'].items():assert sha(z.read(n))==h
        assert sha(Path(__file__).read_bytes())==p['files'][P+Path(__file__).name]
        peer=z.read('results/full-chunk-stage-vendor-optional-fields-verified.json');q=json.loads(peer);assert sha(peer)==p['paired_vendor_recovery_proof_sha256']==w['vendor_recovery_proof_sha256'] and peer==(a.proofs/'full-chunk-stage-vendor-optional-fields-verified.json').read_bytes() and q['original_GPU_archive_sha256']==manifest['linked_vendor_archive_sha256']
        previous=z.read('results/qg128-sdk-control-attempt1-verified.json');previousq=json.loads(previous);assert sha(previous)==p['previous_SDK_failure_proof_sha256']==w['previous_SDK_failure_proof_sha256'] and previous==(a.proofs/'qg128-sdk-control-attempt1-verified.json').read_bytes() and previousq['archive_sha256']==manifest['linked_failed_SDK_archive_sha256'] and previousq['exit_code']==1 and w['driver_module_sha256_before']==w['driver_module_sha256_after']
        for phase in ('dependency','build','SDK'):
            n=base+phase+'-ledger/commands.jsonl'
            if n not in manifest['files']:assert w['exit_code']!=0;continue
            events=[json.loads(line) for line in z.read(n).splitlines()];assert len(events)==2 and events[0]['source']['owned_source_sha256']==events[1]['source_after']['owned_source_sha256']==p['files']
            if w['exit_code']==0:assert events[1]['exit_code']==0
        report=json.loads(z.read(base+'report.json')) if base+'report.json' in manifest['files'] else None
        if w['exit_code']==0:
            assert w['status']=='actual_QG_SDK_control_checked_needs_independent_replay' and sha(z.read(base+'native-build/qg128-sdk-control.exe'))==w['executable_sha256'];r=report;assert r['status']=='QG_M16_SDK_control_M16_M32_M128_compatible' and r['GPU_UUID']=='GPU-265ee66c-9b79-cd3b-0c63-dd6942ddb5e6' and r['device']==0 and r['SM']==120 and r['cublasLt_version']==130400 and r['geometry']==dict(N=12288,K=5120) and r['workspace_limit']==4194304
            assert r['no_model_weights_loaded'] and r['no_matmul_executed'] and r['no_opaque_algorithm_import_or_patch'] and not r['model_or_performance_qualified'];assert 1<=r['heuristic_found']<=8 and 0<=r['first_successful_index']<r['heuristic_found'];assert r['SDK_enums']=={'TILE_64x32':13,'STAGES_64x5':17}
            for k,value in (('ID',67),('TILE_ID',13),('STAGES_ID',17),('CTA_SWIZZLING',1),('CUSTOM_OPTION',18)):assert r['public_configuration'][k]==dict(status=0,bytes=4,value=value)
            assert [c['M'] for c in r['checks']]==[16,32,128] and all(c['API_status']==c['state']==0 and 0<=c['required_workspace']<=4194304 for c in r['checks']);assert set(w['loaded_module_sha256'])=={'cublasLt64_13.dll','nvcuda.dll'} and w['loaded_module_sha256']['cublasLt64_13.dll']==w['runtime_before']['cublasLt64_13.dll'] and w['loaded_module_sha256']['nvcuda.dll']==w['driver_module_sha256_before'];assert (w['optional_cudart_module_sha256']==w['runtime_before']['cudart64_13.dll']) if r['optional_cudart_module_path'] else w['optional_cudart_module_sha256'] is None
        else:assert w['status']=='failed' and w['error']
        result=dict(status='actual_QG_M16_SDK_control_compatibility_replayed' if w['exit_code']==0 else 'actual_QG_SDK_control_failure_preserved',archive_sha256=sha(archive.read_bytes()),archive_bytes=archive.stat().st_size,exit_code=w['exit_code'],error=w.get('error'),vendor_recovery_proof_sha256=sha(peer),previous_SDK_failure_proof_sha256=sha(previous),source_library_runtime_SDK_checkpoint_unchanged=True,report=report,metadata_compatibility_only=True,numerical_equivalence_or_algorithm_cause_or_model_or_performance_qualified=False,two_times_target_achieved=False)
    finally:z.close()
    target=a.proofs/(JOB+'-verified.json');blob=json.dumps(result,indent=2).encode();assert not target.exists() or target.read_bytes()==blob;target.write_bytes(blob);print(json.dumps(dict(status=result['status'],proof_sha256=sha(blob),report=report)))
if __name__=='__main__':main()
