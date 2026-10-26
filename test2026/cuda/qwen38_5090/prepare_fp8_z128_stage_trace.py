"""Pin separate same-DLL initialization trace; failed stage methods remain immutable."""
from pathlib import Path
import argparse,ast,hashlib,json,os,sys,unittest,zipfile
sha=lambda raw:hashlib.sha256(raw).hexdigest();JOB='fp8-z128-stage-witness-attempt1';PREFIX='Garnet/test2026/cuda/qwen38_5090/'
NAMES=('run_fp8_z128_stage_trace.py','test_fp8_z128_stage_trace.bat','prepare_fp8_z128_stage_trace.py','verify_fp8_z128_stage_trace.py','test_fp8_z128_stage_trace.py','evidence/fp8-z128-stage-trace-derivation.json')

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--proofs',type=Path,required=True);parser.add_argument('--work',type=Path,required=True);args=parser.parse_args();m=Path(__file__).resolve().parent;repo=m.parents[2];sys.path[:0]=[str(repo),str(m)]
    proof=(args.proofs/(JOB+'-verified.json')).read_bytes();analysis=(args.proofs/(JOB+'-analysis.json')).read_bytes();q=json.loads(proof);a=json.loads(analysis)
    assert proof==(args.proofs/(JOB+'-remote-verified.json')).read_bytes() and analysis==(args.proofs/(JOB+'-remote-analysis.json')).read_bytes()
    assert q['status']=='fp8_z128_stage_failure_or_remaining_difference_preserved' and a['unchanged_old_T32_baseline_arrays_compared']==200 and not a['T128_profile_built_or_inferred']
    archive=args.proofs/(JOB+'-evidence.zip');assert sha(archive.read_bytes())==q['archive_sha256']
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None;praw=z.read(PREFIX+'evidence/fp8-z128-stage-witness-preparation.json');p=json.loads(praw);w=json.loads(z.read('results/'+JOB+'/workflow.json'))
        assert w['exit_code']==1 and w['cleanup_finished'] and w['baseline_source_restored'] and w['baseline_library_restored']
        assert w['source_before_sha256']==w['source_after_sha256']==p['files'] and len(w['policy_controls'])==10
    p['files'][PREFIX+'evidence/fp8-z128-stage-witness-preparation.json']=sha(praw)
    new={PREFIX+name:sha((m/name).read_bytes()) for name in NAMES}
    for name in NAMES:
        if name.endswith('.py'):ast.parse((m/name).read_bytes())
    for name,h in p['candidate_files'].items():assert sha((repo/name).read_bytes())==h
    p['files'].update(new);p['new_files']=new;p['installed_files']={**p['files'],**{'Garnet/'+name:h for name,h in p['candidate_files'].items()}}
    tests=unittest.defaultTestLoader.loadTestsFromName('test2026.cuda.qwen38_5090.test_fp8_z128_stage_trace')
    def ids(suite):
        for item in suite:
            if isinstance(item,unittest.TestSuite):yield from ids(item)
            else:yield item.id()
    p['test_ids']=sorted(p['test_ids']+list(ids(tests)));assert len(p['test_ids'])==len(set(p['test_ids']))==32
    import run_logged as ledger
    ledger.source_identity=lambda:{'scope':'Explicit trace and candidate pins; unrelated untracked excluded','owned_source_sha256':{**new,**{'Garnet/'+name:h for name,h in p['candidate_files'].items()}}};os.chdir(repo)
    assert ledger.run([sys.executable,'-m','unittest',*p['test_ids']],args.work/'fp8-z128-stage-trace-local-cpu-ledger','correctness','Original29controls plus3trace-isolation/provenance contracts, no local native/GPU',[m/name for name in NAMES])==0
    p['parents'][JOB]={'status':q['status'],'exit_code':1,'proof_sha256':sha(proof),'archive_sha256':q['archive_sha256']}
    p.update(status='fp8_z128_stage_trace_prepared_gpu_unrun',expected_cpu_test_count=32,scope='Same closednative28 DLL and frozen collector/order after failedstage. Skip10alreadyclosed freshpolicycontrols; repeat only main realfirst128stagecollector with NVIDIA cuBLASLt API logging LEVEL5/MASK31 to unique owned log. No cache/selection/math change, no benchmark or model/quality admission. Preserve failures and restore8old/20new source targets, baselineDLL/runtime/SDK/checkpoints/ownedcache.')
    target=m/'evidence/fp8-z128-stage-trace-preparation.json';assert not target.exists();raw=json.dumps(p,indent=2).encode();target.write_bytes(raw)
    payload={name:(repo/name.removeprefix('Garnet/')).read_bytes() for name in new};payload[PREFIX+'evidence/'+target.name]=raw
    # Include already-owned source bundle byte-for-byte; installer permits only identical existing bytes.
    source=args.work/'fp8-z128-native-source.zip';assert sha(source.read_bytes())==p['source_bundle_sha256'];payload[p['source_bundle']]=source.read_bytes()
    package=args.work/'fp8-z128-stage-trace-deployment.zip';assert not package.exists()
    with zipfile.ZipFile(package,'x',zipfile.ZIP_DEFLATED) as z:
        for name,value in payload.items():z.writestr(name,value)
        z.writestr('install-manifest.json',json.dumps({'files':{name:sha(value) for name,value in payload.items()}},indent=2))
    result={'status':p['status'],'files':len(p['files']),'installed':len(p['installed_files']),'targets':len(payload),'CPU_tests':32,'deployment_sha256':sha(package.read_bytes()),'bytes':package.stat().st_size,'preparation_sha256':sha(raw)};(args.work/'fp8-z128-stage-trace-package.json').write_bytes(json.dumps(result,indent=2).encode());print(json.dumps(result))

if __name__=='__main__':main()
