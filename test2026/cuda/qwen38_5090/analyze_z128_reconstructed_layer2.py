"""Replay actual native layer2 arrays, constructor attribution and restored identities."""
from pathlib import Path
import argparse,hashlib,json,sys,zipfile
sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
from test2026.cuda.qwen38_5090.verify_fp8_z128_full4k_model import checked
from test2026.cuda.qwen38_5090.z128_reconstructed_layer2_replay import replay,require_admission
from test2026.cuda.qwen38_5090.z128_reconstructed_layer2_trace import replay as trace_replay
from test2026.cuda.qwen38_5090.modelopt_trace_layer2 import compare_profiles
JOB='z128-reconstructed-layer2-attempt1';OLD='layer2-ordinary-witness-attempt1';M='Garnet/test2026/cuda/qwen38_5090/';sha=lambda b:hashlib.sha256(b).hexdigest()
def main():
    a=argparse.ArgumentParser();a.add_argument('--proofs',type=Path,required=True);args=a.parse_args();archive=args.proofs/(JOB+'-evidence.zip');z,manifest=checked(archive);base='results/'+JOB+'/'
    try:
        doc=lambda name:json.loads(z.read(name));raw=z.read(M+'evidence/z128-reconstructed-layer2-preparation.json');p=json.loads(raw);w=doc(base+'workflow.json');assert w['preparation_sha256']==sha(raw) and type(w['exit_code']) is int
        assert w['cleanup_finished'] is w['baseline_source_restored'] is w['baseline_library_restored'] is True
        assert w['source_before_sha256']==w['source_after_sha256']==p['files'] and w['installed_source_sha256']==p['installed_files']
        assert w['runtime_dependencies']==w['runtime_dependencies_after']==p['runtime'] and w['environment_before']==w['environment_after']==p['environment'] and w['checkpoint_after']==p['checkpoints']
        for name,h in p['new_files'].items():assert sha(z.read(name))==h
        record=json.loads((Path(__file__).resolve().parent/'evidence/z128-reconstructed-layer2-analysis-derivation.json').read_bytes())
        original=z.read(M+record['source']);assert sha(original)==record['source_sha256']==p['new_files'][M+record['source']]
        composed=original
        for c in record['changes']:
            before=c['before'].encode();assert composed.count(before)==c['count'];composed=composed.replace(before,c['after'].encode())
        assert composed==Path(__file__).read_bytes() and sha(composed)==record['target_sha256']
        cpu=[json.loads(line) for line in z.read(base+'cpu-ledger/commands.jsonl').splitlines()];assert len(cpu)==2 and cpu[1]['exit_code']==0 and cpu[0]['source']['owned_source_sha256']==cpu[1]['source_after']['owned_source_sha256']==p['installed_files']
        log=z.read(base+'cpu-ledger/commands.log');assert b'Ran 12 tests' in log and b'Ran 86 tests' in log
        parents={}
        for job,row in p['parents'].items():
            proof=z.read('results/'+job+'-verified.json');q=json.loads(proof);assert sha(proof)==row['proof_sha256'] and proof==(args.proofs/(job+'-verified.json')).read_bytes() and q['archive_sha256']==row['archive_sha256']==sha((args.proofs/(job+'-evidence.zip')).read_bytes())
        oldzip,oldmanifest=checked(args.proofs/(OLD+'-evidence.zip'));parents[OLD]=oldzip;oldbase='results/'+OLD+'/native/';prior=json.loads(oldzip.read(oldbase+'report.json'))
        fixture=z.read(M+'evidence/layer2-ordinary-common-input.f32');assert sha(fixture)==p['common_input_sha256'] and len(fixture)==2621440
        phases={};reports={}
        for phase in ('trace-on','trace-off'):
            reportname=base+phase+'/report.json'
            if reportname not in z.namelist():continue
            r=doc(reportname);reports[phase]=r;assert r['candidate_library_sha256']==p['candidate_library_sha256'] and r['input_ids']==p['common_input_metadata']['input_ids']
            assert r['trace_policy']==r['factory_trace_policy']==('1' if phase=='trace-on' else None)
            ledger=[json.loads(line) for line in z.read(base+phase+'-ledger/commands.jsonl').splitlines()];assert len(ledger)==2 and ledger[0]['source']['owned_source_sha256']==ledger[1]['source_after']['owned_source_sha256']==p['installed_files']
            resource=doc(base+phase+'-resources/resource-monitor.json');assert resource['gpu']['uuid']==p['gpu_uuid'] and resource['child_exit_code']==ledger[1]['exit_code']
            complete=r['status']=='layer2_ordinary_collected_needs_replay';read=lambda name:z.read(base+phase+'/'+name)
            result=replay(r,read,fixture) if complete else None
            if complete:
                assert ledger[1]['exit_code']==0 and doc(base+phase+'-array-analysis.json')==result
            compared=None
            if any(row['tokens']==32 and len(row['repeats'])==2 and all(len(seq)==4 for seq in row['repeats']) for row in r['cases']):
                compared=compare_profiles(r,prior,read,lambda name:oldzip.read(oldbase+name),(32,));assert compared['complete_finite_entries_compared']==58
            traced=trace_replay(z.read(base+phase+'-ledger/commands.log'),phase=='trace-on')
            if base+phase+'-trace-analysis.json' in z.namelist():assert doc(base+phase+'-trace-analysis.json')==json.loads(json.dumps(traced))
            phases[phase]={'collector_status':r['status'],'error':r.get('error'),'arrays':result,'retained_T32':compared,'native_constructor_attribution':traced}
        passed=w['exit_code']==0;paired=None
        if passed:
            assert w['status']=='z128_reconstructed_layer2_collected_needs_replay' and set(phases)=={'trace-on','trace-off'}
            for row in phases.values():require_admission(row['arrays'])
            paired=compare_profiles(reports['trace-on'],reports['trace-off'],lambda name:z.read(base+'trace-on/'+name),lambda name:z.read(base+'trace-off/'+name),(32,128));assert paired['complete_finite_entries_compared']==73
        else:assert w['status']=='failed' and w.get('error')
        result={'status':'actual_reconstructed_Z128_layer2_exact_cross_profile_admission_replayed' if passed else 'actual_reconstructed_Z128_layer2_failure_preserved','archive_sha256':sha(archive.read_bytes()),'archive_bytes':archive.stat().st_size,'archive_members':len(z.namelist()),'candidate_library_sha256':p['candidate_library_sha256'],'exit_code':w['exit_code'],'error':w.get('error'),'source_runtime_SDK_checkpoint_restored':True,'phases':phases,'trace_on_off_array_parity':paired,'layer2_cross_profile_admitted':passed,'full_model_or_quality_or_performance_qualified':False,'two_times_target_achieved':False,'verifier_sha256':sha(Path(__file__).read_bytes()),'original_frozen_verifier_passed':False,'original_frozen_verifier_failure':'JSON trace mapping keys int in memory versus str after serialization; all other checks unchanged','analysis_derivation_sha256':sha((Path(__file__).resolve().parent/'evidence/z128-reconstructed-layer2-analysis-derivation.json').read_bytes())}
        oldzip.close()
    finally:z.close()
    target=args.proofs/(JOB+'-verified.json');raw=json.dumps(result,indent=2).encode();assert not target.exists() or target.read_bytes()==raw;target.write_bytes(raw);print(json.dumps({'status':result['status'],'proof_sha256':sha(raw),'exit_code':w['exit_code']}));return 0
if __name__=='__main__':raise SystemExit(main())
