"""Freeze unchanged runtime and actual parent evidence for an untimed B8 test."""
from pathlib import Path
import argparse,hashlib,json,zipfile
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
METHODS=('run_bf16_batch8_admission.py','prepare_bf16_batch8_admission_run.py','verify_bf16_batch8_admission.py','test_bf16_batch8_admission.bat','probe_bf16_batch8_admission.py','bf16_batch8_admission.py','test_bf16_batch8_admission.py','test_bf16_batch8_owner.py','test_bf16_batch8_replay.py','test_bf16_packed_state_gate.py','test_bf16_low_batch_profile.py','bf16_packed_state_gate.py','bf16_model_state_witness.py','bf16_float_reference_state.py','run_logged.py','run_bf16_low_batch_native_build.py','monitor_native_probe.py','sample_vram.py','evidence/bf16-batch8-admission-offline-preparation.json')
TEST_MODULES=tuple('test2026.cuda.qwen38_5090.'+n for n in ('test_bf16_batch8_admission','test_bf16_packed_state_gate','test_bf16_low_batch_profile','test_bf16_batch8_owner','test_bf16_batch8_replay'))
def main():
 a=argparse.ArgumentParser(description=__doc__);a.add_argument('--proofs',type=Path,required=True);a.add_argument('--plan',type=Path,required=True);args=a.parse_args();m=Path(__file__).resolve().parent;repo=m.parents[2]
 parents={};required=[args.proofs/(n+s) for n in ('fixed-length-http-study-attempt1','tokenizer-nfc-model-consistency-attempt1') for s in ('-verified.json','-remote-verified.json','-evidence.zip')]
 if not all(p.is_file() for p in required):print(json.dumps({'status':'bf16_batch8_parent_proofs_missing_no_preparation'}));return 2
 for name,status in (('fixed-length-http-study-attempt1','fixed_length_http_study_replayed'),('tokenizer-nfc-model-consistency-attempt1','tokenizer_nfc_model_consistency_replayed')):
  proofpath=args.proofs/(name+'-verified.json');q=json.loads(proofpath.read_bytes());assert q==json.loads((args.proofs/(name+'-remote-verified.json')).read_bytes()) and q['status']==status and sha(args.proofs/(name+'-evidence.zip'))==q['archive_sha256']
  with zipfile.ZipFile(args.proofs/(name+'-evidence.zip')) as z:
   closed=json.loads(z.read('results/'+name+'/workflow.json'));assert closed['exit_code']==0 and closed['cleanup_finished'] and closed['library_unchanged'] and closed['source_sha256']==closed['source_after_sha256'] and closed['runtime_dependencies']==closed['runtime_dependencies_after']
   if name=='fixed-length-http-study-attempt1':files=dict(closed['source_sha256']);runtime=closed['runtime_dependencies']
   else:assert closed['runtime_dependencies']==runtime
  parents[name]={'status':status,'proof_sha256':sha(proofpath),'archive_sha256':q['archive_sha256']}
 offline=json.loads((m/'evidence/bf16-batch8-admission-offline-preparation.json').read_bytes());assert sha(m/'probe_bf16_batch8_admission.py')==offline['source_sha256'] and sha(m/'bf16_batch8_admission.py')==offline['helper_sha256']
 for method in METHODS:files['Garnet/test2026/cuda/qwen38_5090/'+method]=sha(m/method)
 for name,digest in files.items():
  if name.startswith('Garnet/'):assert sha(repo/name.removeprefix('Garnet/'))==digest,('Local qualified source drift',name)
 scope='One exclusive untimed existing ordinary B8/BF16KV/T32prefill/T4decode mixedNVFP4/FP8 model admission. Original reference supervisor must first be observed terminal and its exact immutable closed archive supplied; idle GPU/process guard before/after. Unchanged native runtime/checkpoint/frozen inputs;9actual short/full3968+128 text/image cases, rotated and reset-repeat full-row state/component and finite raw hidden/normalized vectors. Record adapter-wide construction/correctness memory and owned engine-cache cleanup. Internal consistency only; independent B1/B4 model-math equivalence,805quality, MTP, HTTP batching, speed and2x remain unqualified.'
 p={'status':'bf16_batch8_model_admission_prepared_gpu_unrun','files':files,'runtime_dependencies':runtime,'parents':parents,'test_modules':TEST_MODULES,'scope':scope};target=m/'evidence/bf16-batch8-admission-run-preparation.json';raw=json.dumps(p,indent=2).encode();assert not target.exists() or target.read_bytes()==raw
 if not target.exists():target.write_bytes(raw)
 uploads=[{'local':str(m/n),'target':'Garnet/test2026/cuda/qwen38_5090/'+n,'sha256':sha(m/n)} for n in METHODS]+[{'local':str(target),'target':'Garnet/test2026/cuda/qwen38_5090/evidence/'+target.name,'sha256':sha(target)}]
 raw=json.dumps({'uploads':uploads,'scope':scope},indent=2).encode();assert not args.plan.exists() or args.plan.read_bytes()==raw
 if not args.plan.exists():args.plan.write_bytes(raw)
 print(json.dumps({'status':p['status'],'preparation_sha256':sha(target),'files':len(files),'upload_targets':len(uploads),'gpu_executed':False}));return 0
if __name__=='__main__':raise SystemExit(main())
