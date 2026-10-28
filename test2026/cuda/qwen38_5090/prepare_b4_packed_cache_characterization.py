"""Prepare packed-only retry after actual full-case and typed empty-state replay."""
from pathlib import Path
import argparse,ast,hashlib,json,os,sys,unittest,zipfile
METHODS=('probe_b4_packed_cache_characterization.py','test_b4_packed_cache_characterization.bat','run_b4_packed_cache_characterization.py',
 'verify_b4_packed_cache_characterization.py','prepare_b4_packed_cache_characterization.py','test_b4_packed_cache_characterization.py','verify_b4_cache_mode_partial.py','evidence/b4-packed-cache-characterization-derivation.json')
sha=lambda raw:hashlib.sha256(raw).hexdigest();C='b4-cache-mode-characterization-attempt1';PREFIX='Garnet/test2026/cuda/qwen38_5090/'

def main():
    a=argparse.ArgumentParser(description=__doc__);a.add_argument('--proofs',type=Path,required=True);a.add_argument('--work',type=Path,required=True);args=a.parse_args()
    m=Path(__file__).resolve().parent;repo=m.parents[2];parents={};base=None;oldp=None
    for job,status,code in (('paged-batch8-model-preservation-attempt1','paged_batch8_model_preservation_replayed',0),('bf16-batch8-external-reference-attempt1','bf16_batch8_external_reference_failure_preserved',1),(C,'b4_cache_mode_characterization_failure_preserved',1)):
        raw=(args.proofs/(job+'-verified.json')).read_bytes();q=json.loads(raw);assert raw==(args.proofs/(job+'-remote-verified.json')).read_bytes() and q['status']==status
        archive=args.proofs/(job+'-evidence.zip');assert sha(archive.read_bytes())==q['archive_sha256']
        with zipfile.ZipFile(archive) as z:
            w=json.loads(z.read('results/'+job+'/workflow.json'));assert type(w['exit_code']) is int and w['exit_code']==code and w['cleanup_finished'] is True
            assert w['source_sha256']==w['source_after_sha256'] and w['runtime_dependencies']==w['runtime_dependencies_after'] and w['environment_before']==w['environment_after']
            if job==C:
                base=w;oldraw=z.read(PREFIX+'evidence/b4-cache-mode-characterization-preparation.json');oldp=json.loads(oldraw)
        parents[job]={'status':status,'exit_code':code,'proof_sha256':sha(raw),'archive_sha256':q['archive_sha256']}
    partial_path=args.proofs/(C+'-partial-verified.json');partial_raw=partial_path.read_bytes();partial=json.loads(partial_raw)
    assert partial_raw==(args.proofs/(C+'-partial-remote-verified.json')).read_bytes() and partial['status']=='b4_cache_mode_partial_full_and_empty_state_replayed'
    assert partial['archive_sha256']==parents[C]['archive_sha256'] and partial['packed_cases_completed']==0
    assert partial['full_repeat_vs_failed_full']['all_observed_ids_states_vectors_exact'] is True and partial['packed_empty_state']['all_states_and_vectors_exact'] is True
    new={PREFIX+name:sha((m/name).read_bytes()) for name in METHODS};assert partial['external_replay_verifier_sha256']==new[PREFIX+'verify_b4_cache_mode_partial.py']
    files={**base['source_after_sha256'],PREFIX+'evidence/b4-cache-mode-characterization-preparation.json':sha(oldraw),'results/'+C+'-partial-verified.json':sha(partial_raw)}
    assert not set(files)&set(new);files.update(new)
    for row in json.loads((m/'evidence/b4-packed-cache-characterization-derivation.json').read_bytes())['files']:
        raw=(m/row['source']).read_bytes();assert sha(raw)==row['source_sha256'];text=raw.decode()
        for step in row['replacements']:
            assert text.count(step['old'])==step['count'];text=text.replace(step['old'],step['new'])
        assert text.encode()==(m/row['target']).read_bytes() and sha(text.encode())==row['target_sha256']
    for name in METHODS:
        if name.endswith('.py'):ast.parse((m/name).read_bytes())
    sys.path[:0]=[str(repo),str(m)]
    suite=unittest.defaultTestLoader.loadTestsFromName('test2026.cuda.qwen38_5090.test_b4_packed_cache_characterization')
    def ids(suite):
        for test in suite:
            if isinstance(test,unittest.TestSuite):yield from ids(test)
            else:yield test.id()
    tests=sorted(oldp['test_ids']+list(ids(suite)));assert len(set(tests))==len(tests)==53
    import test2026.cuda.qwen38_5090.run_logged as ledger
    ledger.source_identity=lambda:{'scope':'Explicit owned packed retry controls; unrelated untracked files excluded','owned_source_sha256':new}
    os.chdir(repo);code=ledger.run([sys.executable,'-m','unittest',*tests],args.work/'b4-packed-cache-characterization-bound-cpu-ledger','correctness','Previous50controls plus3typed empty-state and partial-proof archive dependency controls; original full-reference gate unchanged; no local runtime/GPU',[m/name for name in METHODS]);assert code==0
    p={'status':'b4_packed_cache_characterization_prepared_gpu_unrun','files':files,'new_files':new,'runtime':base['runtime_dependencies_after'],'environment':base['environment_after'],'parents':parents,
       'partial_parent':{'job':C,'proof_sha256':sha(partial_raw),'archive_sha256':partial['archive_sha256']},'test_ids':tests,'expected_cpu_test_count':53,'gpu_uuid':oldp['gpu_uuid'],
       'scope':'Only the actual first four-row packed-BF16 B4 short8-output case, after original dual-mode owner failed at a full-only representation control before any packed case. Original failed archive and strict gate unchanged. Correct packed empty/final-reset control requires exact full expanded-state hashes and raw vectors against B8 packed empties. Reuse independently replayed actual full companion (which exactly reproduces prior failed-full) for same-B4 comparison; record every ID/state/vector difference versus full and B8. No tolerance relaxation, cause/harmlessness/semantic/model-math/quality/performance admission. Existing1a1d source/runtime/request recipe, exclusive owner/source/SDK/cache cleanup and both-machine replay required. LocalT128native candidate remains uninstalled.'}
    target=m/'evidence/b4-packed-cache-characterization-preparation.json';raw=json.dumps(p,indent=2).encode();assert not target.exists() or target.read_bytes()==raw
    if not target.exists():target.write_bytes(raw)
    rows=[{'local':str(m/name),'target':PREFIX+name,'sha256':sha((m/name).read_bytes())} for name in METHODS];rows.append({'local':str(target),'target':PREFIX+'evidence/'+target.name,'sha256':sha(raw)})
    (args.work/'b4-packed-cache-characterization-upload-plan.json').write_bytes(json.dumps({'uploads':rows,'scope':p['scope']},indent=2).encode())
    print(json.dumps({'status':p['status'],'source_pins':len(files),'cpu_tests':53,'preparation_sha256':sha(raw)}));return 0

if __name__=='__main__':raise SystemExit(main())
