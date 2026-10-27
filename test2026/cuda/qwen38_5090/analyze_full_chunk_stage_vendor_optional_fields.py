"""CPU-only recovery of complete logged captures after an optional-field reader failure."""
from pathlib import Path
import argparse,hashlib,json,sys,traceback,zipfile
sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
from test2026.cuda.qwen38_5090.verify_gdn_chunk_stage_witness_bound import checked
from test2026.cuda.qwen38_5090 import full_chunk_stage_vendor as original
from test2026.cuda.qwen38_5090 import full_chunk_stage_vendor_optional_fields as fixed
JOB='full-chunk-stage-vendor-attempt1';P='Garnet/test2026/cuda/qwen38_5090/';sha=lambda b:hashlib.sha256(b).hexdigest()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--proofs',type=Path,required=True);args=ap.parse_args();M=Path(__file__).resolve().parent
    qraw=(args.proofs/(JOB+'-verified.json')).read_bytes();q=json.loads(qraw);assert q['status']=='actual_full_chunk_vendor_logging_failure_preserved' and q['exit_code']==1 and q['source_library_runtime_SDK_checkpoint_restored']
    archive=args.proofs/(JOB+'-evidence.zip');assert sha(archive.read_bytes())==q['archive_sha256'];z,manifest=checked(archive);base='results/'+JOB+'/'
    try:
        p=json.loads(z.read(P+'evidence/full-chunk-stage-vendor-preparation.json'));w=json.loads(z.read(base+'workflow.json'));assert w['exit_code']==1 and w['cleanup_finished'] and w['error']=='AssertionError()' and w['fresh_logged_profiles']==['stages32','stages128']
        assert w['inherited_defaultoff_controls']=={'parent':'full-chunk-stage-attempt2','profiles':['control32','control128']}
        for phase in ('stages32','stages128'):
            events=[json.loads(b) for b in z.read(base+phase+'-ledger/commands.jsonl').splitlines()];assert len(events)==2 and events[1]['exit_code']==0
        assert sha(Path(original.__file__).read_bytes())==p['new_files'][P+'full_chunk_stage_vendor.py']
        derivation=(M/'evidence/full-chunk-stage-vendor-optional-fields-derivation.json').read_bytes();d=json.loads(derivation);raw=Path(original.__file__).read_bytes();assert sha(raw)==d['source_sha256'] and len(d['changes'])==1
        for c in d['changes']:assert raw.count(c['old'].encode())==c['count'];raw=raw.replace(c['old'].encode(),c['new'].encode())
        assert raw==Path(fixed.__file__).read_bytes() and sha(raw)==d['target_sha256'];assert d['original_vendor_archive_sha256']==q['archive_sha256'] and d['numeric_primary_state_vector_token_readonly_repeat_array_predicates_unchanged'] and not d['new_GPU_run']
        parent='full-chunk-stage-attempt2';parentraw=(args.proofs/(parent+'-verified.json')).read_bytes();parentq=json.loads(parentraw);assert parentq['exit_code']==0 and sha(parentraw)==p['parents'][parent]['proof_sha256']
        priorarchive=args.proofs/(parent+'-evidence.zip');assert sha(priorarchive.read_bytes())==p['parents'][parent]['archive_sha256']==parentq['archive_sha256']
        with zipfile.ZipFile(priorarchive) as old:
            prior={};oldreads={};parentbase='results/'+parent+'/'
            for phase in ('control32','control128','stages32','stages128'):prior[phase]=json.loads(old.read(parentbase+phase+'/report.json'));oldreads[phase]=lambda n,phase=phase:old.read(parentbase+phase+'/'+n)
            reports={phase:json.loads(z.read(base+phase+'/report.json')) for phase in ('stages32','stages128')};reads={phase:lambda n,phase=phase:z.read(base+phase+'/'+n) for phase in reports};logs={phase:z.read(base+phase+'-cublaslt.log') for phase in reports}
            try:original.compare(reports,reads,prior,oldreads,args.proofs,logs)
            except AssertionError as e:
                tb=traceback.extract_tb(e.__traceback__);assert tb[-1].name=='parse_qg' and Path(tb[-1].filename).name=='full_chunk_stage_vendor.py';failure=[dict(file=Path(t.filename).name,line=t.lineno,function=t.name,code=t.line) for t in tb]
            else:raise AssertionError('Original optional-field parser failure not reproduced')
            observed=fixed.compare(reports,reads,prior,oldreads,args.proofs,logs);assert observed['arrays']==parentq['arrays']
        result=dict(status='actual_complete_logged_stages_recovered_CPU_optional_vendor_field_reader_only',original_GPU_archive_sha256=q['archive_sha256'],original_GPU_owner_exit_code=1,original_failure_proof_sha256=sha(qraw),original_reader_failure_reproduced=failure,reader_derivation_sha256=sha(derivation),fixed_reader_sha256=d['target_sha256'],analyzer_sha256=sha(Path(__file__).read_bytes()),paired_unlogged_stage_proof_sha256=sha(parentraw),observations=observed,no_new_GPU_run=True,source_library_runtime_SDK_checkpoint_restored=True,algorithm_intervention_tested=False,model_or_performance_qualified=False,two_times_target_achieved=False)
    finally:z.close()
    target=args.proofs/'full-chunk-stage-vendor-optional-fields-verified.json';blob=json.dumps(result,indent=2).encode();assert not target.exists() or target.read_bytes()==blob;target.write_bytes(blob);print(json.dumps(dict(status=result['status'],proof_sha256=sha(blob),vendor={n:r['unique_printed_configurations'] for n,r in observed['vendor'].items()},QG_weight_metadata=observed['QG_weight_metadata'])));return 0
if __name__=='__main__':raise SystemExit(main())
