"""Freeze full4K admission against actual native and T16 short-model evidence."""
from pathlib import Path
import argparse,hashlib,json,zipfile
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
METHODS=('mtp_prefill16_full4k.py','probe_mtp_prefill16_full4k.py','test_mtp_prefill16_full4k.py','run_mtp_prefill16_full4k.py','test_mtp_prefill16_full4k_owner.py','prepare_mtp_prefill16_full4k_run.py','verify_mtp_prefill16_full4k.py','test_mtp_prefill16_full4k.bat','evidence/native-host-stage-cases.json','evidence/native-host-stage-worker-derivation.json','prepare_native_host_stage_cases.py')


def main():
    a=argparse.ArgumentParser(description=__doc__);a.add_argument('--proofs',type=Path,required=True);a.add_argument('--plan',type=Path,required=True);args=a.parse_args();m=Path(__file__).resolve().parent;repo=m.parents[2];parents={}
    names=('fixed-length-http-study-attempt1','mtp-prefill16-model-consistency-attempt1');statuses=('fixed_length_http_study_replayed','mtp_prefill16_short_target_model_replayed')
    for name,status in zip(names,statuses):
        proofpath=args.proofs/(name+'-verified.json');proof=json.loads(proofpath.read_bytes());assert proof==json.loads((args.proofs/(name+'-remote-verified.json')).read_bytes()) and proof['status']==status and sha(args.proofs/(name+'-evidence.zip'))==proof['archive_sha256']
        with zipfile.ZipFile(args.proofs/(name+'-evidence.zip')) as z:
            closed=json.loads(z.read('results/'+name+'/workflow.json'));assert closed['exit_code']==0 and closed['cleanup_finished'] and closed['library_unchanged'] and closed['source_sha256']==closed['source_after_sha256'] and closed['runtime_dependencies']==closed['runtime_dependencies_after']
            if name==names[0]:runtime=closed['runtime_dependencies']
            else:
                assert runtime==closed['runtime_dependencies'];files=dict(closed['source_sha256']);old_preparation=json.loads(z.read('Garnet/test2026/cuda/qwen38_5090/evidence/mtp-prefill16-model-run-preparation.json'))
        parents[name]={'status':status,'proof_sha256':sha(proofpath),'archive_sha256':proof['archive_sha256']}
    test_modules=old_preparation['test_modules']+['test2026.cuda.qwen38_5090.test_mtp_prefill16_full4k','test2026.cuda.qwen38_5090.test_mtp_prefill16_full4k_owner']
    for name in METHODS:files['Garnet/test2026/cuda/qwen38_5090/'+name]=sha(m/name)
    for name,digest in files.items():
        if name.startswith('Garnet/'):assert sha(repo/name.removeprefix('Garnet/'))==digest,('Qualified native source drift',name)
    cases=json.loads((m/'evidence/native-host-stage-cases.json').read_bytes());assert cases['parent_archive_sha256']==parents[names[0]]['archive_sha256'] and cases['parent_proof_sha256']==parents[names[0]]['proof_sha256'] and len(cases['cases'])==4
    scope='One exclusive actual B1 T16draft-prefill full4K correctness job after original reference characterization and B8 supervisors are observed terminal, exact immutable closure/proofs supplied and adapter/process idle. Existing1a1d target/runtime/T4decode/verifier unchanged. Six3968+128 retained text/image inputs including first-text/image repeats; ordinary and candidate complete target state/raw-vector equality at prefill/final, actual full prompt/output/callback IDs and249draft-prefill calls required. Extra resident engine memory sampled through construction and all cases, owned new caches retired. No805quality, HTTP or speed/2x admission.'
    result={'status':'mtp_prefill16_full4k_prepared_gpu_unrun','files':files,'runtime_dependencies':runtime,'parents':parents,'test_modules':test_modules,'scope':scope};target=m/'evidence/mtp-prefill16-full4k-run-preparation.json';raw=json.dumps(result,indent=2).encode();assert not target.exists() or target.read_bytes()==raw
    if not target.exists():target.write_bytes(raw)
    uploads=[{'local':str(m/n),'target':'Garnet/test2026/cuda/qwen38_5090/'+n,'sha256':sha(m/n)} for n in METHODS]+[{'local':str(target),'target':'Garnet/test2026/cuda/qwen38_5090/evidence/'+target.name,'sha256':sha(target)}];raw=json.dumps({'uploads':uploads,'scope':scope},indent=2).encode();assert not args.plan.exists() or args.plan.read_bytes()==raw
    if not args.plan.exists():args.plan.write_bytes(raw)
    print(json.dumps({'status':result['status'],'preparation_sha256':sha(target),'pinned_files':len(files),'upload_targets':len(uploads),'gpu_executed':False}));return 0


if __name__=='__main__':raise SystemExit(main())
