"""Freeze output/isolation diagnosis without weakening failed benchmark gates."""
from pathlib import Path
import argparse,hashlib,json,zipfile
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
METHODS=('run_sglang_output_characterization.py','prepare_sglang_output_characterization_run.py','verify_sglang_output_characterization.py','reference_concurrency_characterization.py','replay_reference_concurrency_characterization.py','test_reference_concurrency_characterization.py','run_mtp_prefill16_model.py','run_bf16_batch8_admission.py','test_mtp_prefill16_model_owner.py','test_bf16_batch8_owner.py','evidence/sglang-output-characterization-replayer-derivation.json')
TEST_MODULES=tuple('test2026.cuda.qwen38_5090.'+n for n in ('test_reference_concurrency_characterization','test_fixed_length_reference','test_reference_mtp_equivalence','test_reference_fixed4k_identity','test_sglang_fixed4k_owner','test_mtp_prefill16_model_owner','test_bf16_batch8_owner'))
def main():
 a=argparse.ArgumentParser(description=__doc__);a.add_argument('--proofs',type=Path,required=True);a.add_argument('--plan',type=Path,required=True);args=a.parse_args();m=Path(__file__).resolve().parent;repo=m.parents[2];prior='sglang-fixed4k-ordinary-attempt2';native='fixed-length-http-study-attempt1'
 required=[args.proofs/(name+suffix) for name in (prior,native) for suffix in ('-verified.json','-remote-verified.json','-evidence.zip')]+[args.proofs/(prior+suffix) for suffix in ('-partial-verified.json','-partial-remote-verified.json')]
 if not all(p.is_file() for p in required):print(json.dumps({'status':'sglang_characterization_prior_proofs_missing_no_preparation'}));return 2
 q=json.loads((args.proofs/(prior+'-verified.json')).read_bytes());partial=json.loads((args.proofs/(prior+'-partial-verified.json')).read_bytes());parent=json.loads((args.proofs/(native+'-verified.json')).read_bytes())
 assert q==json.loads((args.proofs/(prior+'-remote-verified.json')).read_bytes()) and q['status']=='reference_fixed4k_failure_preserved'
 assert partial==json.loads((args.proofs/(prior+'-partial-remote-verified.json')).read_bytes()) and partial['status']=='reference_fixed4k_attempt2_failure_partial_evidence_replayed' and sha(args.proofs/(prior+'-evidence.zip'))==q['archive_sha256']==partial['archive_sha256']
 assert parent==json.loads((args.proofs/(native+'-remote-verified.json')).read_bytes()) and parent['status']=='fixed_length_http_study_replayed' and parent['native_fixed_4k_metrics_replayed'] and sha(args.proofs/(native+'-evidence.zip'))==parent['archive_sha256']
 with zipfile.ZipFile(args.proofs/(prior+'-evidence.zip')) as z:
  w=json.loads(z.read('results/'+prior+'/workflow.json'));assert w['exit_code']==1 and w['cleanup_finished'] and w['source_sha256']==w['source_after_sha256'] and w['runtime_dependencies']==w['runtime_dependencies_after'];files=dict(w['source_sha256']);runtime=w['runtime_dependencies']
 for name in METHODS:files['Garnet/test2026/cuda/qwen38_5090/'+name]=sha(m/name)
 files['results/'+prior+'-verified.json']=sha(args.proofs/(prior+'-verified.json'));files['results/'+prior+'-partial-verified.json']=sha(args.proofs/(prior+'-partial-verified.json'))
 for name,digest in files.items():
  if name.startswith('Garnet/'):assert sha(repo/name.removeprefix('Garnet/'))==digest,('Qualified source drift',name)
 scope='One owned ordinary SGLang0.5.22 output/isolation diagnosis after actual B8 supervisor terminal/closed archive and idle GPU/process guards. Existing configured engine4098/two reserved slots exposes unchanged3968+128 actual full4K inputs, BF16KV/F32SSM/cache-off/graph controls unchanged.192actual nonstream IDs and96streams in135observed-overlap waves: serial before/after, concurrency1/2/4/8, rotated requests, full text/image and mixed-color/spatial cases including32short color streams with valid burst arrivals. Preserve all variants, never infer streamed IDs or a harmless numeric cause. FourKi controls retain strict lengths/IDs/usage. No timing/throughput summary, physical GPU batch, broad quality or2x admission; prior strict failures stay immutable.'
 p={'status':'sglang_output_characterization_prepared_unrun','files':files,'runtime_dependencies':runtime,'native_proof_sha256':sha(args.proofs/(native+'-verified.json')),'native_archive_sha256':parent['archive_sha256'],'prior_partial_proof_sha256':sha(args.proofs/(prior+'-partial-verified.json')),'prior_archive_sha256':q['archive_sha256'],'test_modules':TEST_MODULES,'scope':scope};target=m/'evidence/sglang-output-characterization-run-preparation.json';raw=json.dumps(p,indent=2).encode();assert not target.exists() or target.read_bytes()==raw
 if not target.exists():target.write_bytes(raw)
 uploads=[{'local':str(m/name),'target':'Garnet/test2026/cuda/qwen38_5090/'+name,'sha256':sha(m/name)} for name in METHODS]+[{'local':str(target),'target':'Garnet/test2026/cuda/qwen38_5090/evidence/'+target.name,'sha256':sha(target)}];raw=json.dumps({'uploads':uploads,'scope':scope},indent=2).encode();assert not args.plan.exists() or args.plan.read_bytes()==raw
 if not args.plan.exists():args.plan.write_bytes(raw)
 print(json.dumps({'status':p['status'],'preparation_sha256':sha(target),'files':len(files),'upload_targets':len(uploads),'remote_or_gpu_run':False}));return 0
if __name__=='__main__':raise SystemExit(main())
