"""Freeze explicit prefill16 candidate against the actual unchanged native proofs."""
from pathlib import Path
import argparse,hashlib,json,zipfile
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
METHODS=('run_mtp_prefill16_model.py','prepare_mtp_prefill16_model_run.py','verify_mtp_prefill16_model.py','probe_mtp_prefill16_model.py','test_mtp_prefill16_model.bat','test_mtp_prefill16_study.py','test_mtp_prefill16_model_owner.py','test_mtp_builder_workspace.py','run_bf16_batch8_admission.py','test_bf16_batch8_owner.py','compare_tokenizer_model_parent.py','test_tokenizer_model_parent.py','probe_bf16_low_batch_model_fixed.py','test_bf16_packed_state_gate.py','test_bf16_model_state_witness.py','test_bf16_same_precision_gate.py','test_bf16_padded_state_gate.py','evidence/mtp-prefill16-study-offline-preparation.json','evidence/mtp-prefill16-owner-derivation.json')
TEST_MODULES=tuple('test2026.cuda.qwen38_5090.'+n for n in ('test_mtp_prefill16_study','test_mtp_builder_workspace','test_bf16_batch8_owner','test_mtp_prefill16_model_owner','test_bf16_packed_state_gate','test_bf16_model_state_witness','test_bf16_same_precision_gate','test_bf16_padded_state_gate','test_tokenizer_model_parent'))
def main():
 a=argparse.ArgumentParser(description=__doc__);a.add_argument('--proofs',type=Path,required=True);a.add_argument('--plan',type=Path,required=True);args=a.parse_args();m=Path(__file__).resolve().parent;repo=m.parents[2]
 names=('fixed-length-http-study-attempt1','tokenizer-nfc-model-consistency-attempt1');required=[args.proofs/(n+s) for n in names for s in ('-verified.json','-remote-verified.json','-evidence.zip')]
 if not all(p.is_file() for p in required):print(json.dumps({'status':'mtp_prefill16_native_proofs_missing_no_preparation'}));return 2
 parents={}
 for name,status in zip(names,('fixed_length_http_study_replayed','tokenizer_nfc_model_consistency_replayed')):
  proofpath=args.proofs/(name+'-verified.json');proof=json.loads(proofpath.read_bytes());assert proof==json.loads((args.proofs/(name+'-remote-verified.json')).read_bytes()) and proof['status']==status and sha(args.proofs/(name+'-evidence.zip'))==proof['archive_sha256']
  with zipfile.ZipFile(args.proofs/(name+'-evidence.zip')) as z:
   closed=json.loads(z.read('results/'+name+'/workflow.json'));assert closed['exit_code']==0 and closed['cleanup_finished'] and closed['library_unchanged'] and closed['source_sha256']==closed['source_after_sha256'] and closed['runtime_dependencies']==closed['runtime_dependencies_after']
   if name==names[0]:files=dict(closed['source_sha256']);runtime=closed['runtime_dependencies']
   else:
    assert closed['runtime_dependencies']==runtime
    for path in z.namelist():
     if path.startswith('results/'+name+'/b1/'):files[path]=hashlib.sha256(z.read(path)).hexdigest()
  parents[name]={'status':status,'proof_sha256':sha(proofpath),'archive_sha256':proof['archive_sha256']}
 candidate=repo/'xModel/qwen35/vl_0_8b/cuda_mtp_prefill16_study.py';offline=json.loads((m/'evidence/mtp-prefill16-study-offline-preparation.json').read_bytes())
 assert sha(candidate)==offline['module_sha256'] and sha(m/'probe_mtp_prefill16_model.py')==offline['probe_sha256'] and sha(m/'test_mtp_prefill16_study.py')==offline['test_sha256'] and not offline['gpu_model_executed']
 for name in METHODS:files['Garnet/test2026/cuda/qwen38_5090/'+name]=sha(m/name)
 files['Garnet/xModel/qwen35/vl_0_8b/cuda_mtp_prefill16_study.py']=sha(candidate)
 for name,digest in files.items():
  if name.startswith('Garnet/'):assert sha(repo/name.removeprefix('Garnet/'))==digest,('Qualified native source drift',name)
 scope='One exclusive explicit prefill16 short-model admission after original B8 and SGLang supervisors are observed terminal and exact immutable closed archives supplied. Existing1a1d target source/runtime unchanged; new extra T16 draft engine shares explicit draft KV and retains original T4 decode/verifier, with extra resident weights/buffers fully reported. Run unchanged real8B1 ordinary/MTP/text/image/color/reset/prefix/rejection probe, compare every target state/raw vector/ID/callback to retained current1a1d B1, verify actual call counts and sample construction/correctness adapter memory. Reuse and retire only newly owned engine caches. Actual4Kmemory/model,805quality, HTTP/repeated speed, strongest refs/INT4/4B and2x remain open.'
 p={'status':'mtp_prefill16_model_prepared_gpu_unrun','files':files,'runtime_dependencies':runtime,'parents':parents,'test_modules':TEST_MODULES,'scope':scope};target=m/'evidence/mtp-prefill16-model-run-preparation.json';raw=json.dumps(p,indent=2).encode();assert not target.exists() or target.read_bytes()==raw
 if not target.exists():target.write_bytes(raw)
 uploads=[{'local':str(m/n),'target':'Garnet/test2026/cuda/qwen38_5090/'+n,'sha256':sha(m/n)} for n in METHODS]+[{'local':str(candidate),'target':'Garnet/xModel/qwen35/vl_0_8b/cuda_mtp_prefill16_study.py','sha256':sha(candidate)},{'local':str(target),'target':'Garnet/test2026/cuda/qwen38_5090/evidence/'+target.name,'sha256':sha(target)}]
 raw=json.dumps({'uploads':uploads,'scope':scope},indent=2).encode();assert not args.plan.exists() or args.plan.read_bytes()==raw
 if not args.plan.exists():args.plan.write_bytes(raw)
 print(json.dumps({'status':p['status'],'preparation_sha256':sha(target),'files':len(files),'upload_targets':len(uploads),'gpu_executed':False}));return 0
if __name__=='__main__':raise SystemExit(main())
