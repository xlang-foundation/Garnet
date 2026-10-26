"""Replay actual complete layer3 KV captures and unchanged retained ordinary controls."""
from pathlib import Path
import argparse,hashlib,json,sys,zipfile
sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
from test2026.cuda.qwen38_5090.verify_fp8_z128_full4k_model import checked
from test2026.cuda.qwen38_5090.fa3_prefix_cache_witness import PREFIXES,read_cache,same_state,cache_difference,check_graphs
JOB='fa3-prefix-cache-witness-attempt1';M='Garnet/test2026/cuda/qwen38_5090/';sha=lambda raw:hashlib.sha256(raw).hexdigest()
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--proofs',type=Path,required=True);a=parser.parse_args();archive=a.proofs/(JOB+'-evidence.zip');z,manifest=checked(archive);parents={}
    try:
        base='results/'+JOB+'/';praw=z.read(M+'evidence/fa3-prefix-cache-witness-preparation.json');p=json.loads(praw);w=json.loads(z.read(base+'workflow.json'))
        assert sha(praw)==w['preparation_sha256'] and w['cleanup_finished'] and w['baseline_source_restored'] and w['baseline_library_restored'] and not w['performance_admitted']
        assert w['source_before_sha256']==w['source_after_sha256']==p['files'] and w['runtime_dependencies']==w['runtime_dependencies_after']==p['runtime'] and w['environment_before']==w['environment_after']==p['environment'] and w['checkpoint_after']==p['checkpoints']
        assert w['installed_source_sha256']==p['installed_files'] and len(p['candidate_files'])==35 and w['candidate_files']==p['candidate_files'] and w['candidate_base']==p['candidate_base']
        for name,h in {**p['new_files'],**p['checkpoints'],**p['audit_dependencies']}.items():assert sha(z.read(name))==h
        for name in ('fa3_prefix_cache_witness.py','verify_fa3_prefix_cache_witness.py','bf16_packed_state_gate.py'):
            assert sha(Path(__file__).with_name(name).read_bytes())==p['files'][M+name]
        for job,row in p['parents'].items():
            proof=z.read('results/'+job+'-verified.json');q=json.loads(proof);assert sha(proof)==row['proof_sha256'] and q['status']==row['status'] and q['archive_sha256']==row['archive_sha256']==manifest['linked_parent_archives'][job]
            assert proof==(a.proofs/(job+'-verified.json')).read_bytes() and sha((a.proofs/(job+'-evidence.zip')).read_bytes())==row['archive_sha256'];parents[job]=checked(a.proofs/(job+'-evidence.zip'))[0]
        assert b'Ran 74 tests' in z.read(base+'cpu-ledger/commands.log') and b'\nOK' in z.read(base+'cpu-ledger/commands.log')
        for phase in ('cpu','baseline','native'):
            name=base+phase+'-ledger/commands.jsonl'
            if name not in manifest['files']:continue
            events=[json.loads(raw) for raw in z.read(name).splitlines()];assert len(events)==2 and events[0]['event']=='started' and events[1]['event']=='finished' and events[0]['source']['owned_source_sha256']==events[1]['source_after']['owned_source_sha256']==p['installed_files']
            if w['exit_code']==0:assert events[1]['exit_code']==0
            if phase!='cpu':
                resource=json.loads(z.read(base+phase+'-resources/resource-monitor.json'));assert resource['gpu']['uuid']==p['gpu_uuid']
                if w['exit_code']==0:assert resource['child_exit_code']==0
        comparison=[];reports={}
        if w['exit_code']==0:
            for phase,tokens,parent in (('baseline',32,'mtp-prefill16-full4k-admission-attempt1'),('native',128,'fp8-z128-full4k-model-admission-attempt1')):
                current=base+phase+'/';r=json.loads(z.read(current+'report.json'));assert r['status']=='fa3_prefix_caches_collected_retained_ordinary_control_exact' and r['tokens']==tokens and not r['performance_admitted'] and not r['cause_established'] and r['candidate_library_sha256']==p['candidate_library_sha256']
                oldbase='results/'+parent+'/native/';old=parents[parent];ref=json.loads(old.read(oldbase+'report.json'));record=ref['cases'][0] if tokens==32 else ref['ordinary_observations'][0];read=lambda name:z.read(current+name);oldread=lambda name:old.read(oldbase+name)
                check_graphs(r['graphs'],tokens);assert r['prompt_ids']==record['prompt_ids'] and len(r['prompt_ids'])==3968 and r['ordinary_ids']==record['ordinary_ids'] and len(r['ordinary_ids'])==128
                same_state(r['empty_state'],ref['empty_state'],read,oldread)
                for name in ('ordinary_prefill','ordinary_final'):same_state(r[name],record[name],read,oldread)
                assert [row['prefix'] for row in r['snapshots']]==list(PREFIXES)
                for row in r['snapshots']:assert row['slots']==[row['prefix']] and set(row['caches'])=={'key','value'}
                reports[phase]=r
            for x,y in zip(reports['native']['snapshots'],reports['baseline']['snapshots']):
                assert x['prefix']==y['prefix'];row={'prefix':x['prefix'],'caches':{}}
                for name in ('key','value'):
                    actual=read_cache(x['caches'][name],lambda file:z.read(base+'native/'+file));reference=read_cache(y['caches'][name],lambda file:z.read(base+'baseline/'+file));row['caches'][name]=cache_difference(actual,reference,x['prefix'])
                comparison.append(row)
        passed=w['exit_code']==0
        result={'status':'fa3_prefix_cache_controls_replayed' if passed else 'fa3_prefix_cache_failure_preserved','archive_sha256':sha(archive.read_bytes()),'files_verified':len(manifest['files']),'source_runtime_SDK_checkpoint_restored':True,'retained_ordinary_controls_exact':passed,'prefixes':comparison,'first_sampled_different_prefix':next((row['prefix'] for row in comparison if any(not value['exact'] for value in row['caches'].values())),None),'cause_established':False,'performance_admitted':False,'two_times_target_achieved':False,'verifier_sha256':sha(Path(__file__).read_bytes()),'scope':p['scope']}
    finally:
        z.close()
        for value in parents.values():value.close()
    raw=json.dumps(result,indent=2).encode();target=a.proofs/(JOB+'-verified.json');assert not target.exists() or target.read_bytes()==raw
    if not target.exists():target.write_bytes(raw)
    print(json.dumps({'status':result['status'],'proof_sha256':sha(raw),'first_sampled_different_prefix':result['first_sampled_different_prefix']}));return 0 if passed else 1
if __name__=='__main__':raise SystemExit(main())
