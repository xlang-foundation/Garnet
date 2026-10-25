"""Pin the small actual EOS-policy regression from qualified model and preserved failure."""
from pathlib import Path
import argparse,hashlib,json,zipfile
sha=lambda raw:hashlib.sha256(raw).hexdigest()
METHODS=('prepare_fixed_length_eos.py','run_fixed_length_eos.py','probe_fixed_length_eos.py','fixed_length_policy.py','native_fixed_length_stream_worker.py','native_fixed_length_http_bridge.py','serve_native_fixed_length_worker.bat','benchmark_fixed_length_native.py','benchmark_fixed_length_reference.py','test_fixed_length_worker_contract.py','test_fixed_length_native_stream.py','test_fixed_length_reference.py','benchmark_stream.py','metrics.py','sample_vram.py','probe_reference_http.py','run_logged.py','monitor_native_probe.py','run_bf16_low_batch_native_build.py','vllm_cache_control.py','test_mmmu_http_quality.py','test_mmmu_quality_protocol.py','mmmu_http_quality.py','mmmu_quality_protocol.py','verify_mmmu_http_quality.py','evidence/fixed-length-4096-corpus.zip','evidence/fixed-length-worker-derivation.json','evidence/fixed-length-clients-derivation.json','evidence/native-natural-eos-failure-reanalysis.json')

def main():
 a=argparse.ArgumentParser(description=__doc__);a.add_argument('--proofs',type=Path,required=True);a.add_argument('--plan',type=Path,required=True);args=a.parse_args();m=Path(__file__).resolve().parent;repo=m.parents[2];model='tokenizer-nfc-model-consistency-attempt1';failure='tokenizer-nfc-http-study-attempt1';parents={}
 for name,status in ((model,'tokenizer_nfc_model_consistency_replayed'),(failure,'tokenizer_nfc_http_study_failure_preserved')):
  raw=(args.proofs/(name+'-remote-verified.json')).read_bytes();proof=json.loads(raw);assert proof==json.loads((args.proofs/(name+'-verified.json')).read_bytes()) and proof['status']==status and sha((args.proofs/(name+'-evidence.zip')).read_bytes())==proof['archive_sha256'];parents[name]={'proof_sha256':sha(raw),'archive_sha256':proof['archive_sha256'],'status':status}
 with zipfile.ZipFile(args.proofs/(model+'-evidence.zip')) as z:
  w=json.loads(z.read('results/'+model+'/workflow.json'));assert w['cleanup_finished'] and w['exit_code']==0 and w['source_sha256']==w['source_after_sha256'] and w['runtime_dependencies']==w['runtime_dependencies_after'];files=dict(w['source_sha256']);runtime=w['runtime_dependencies']
 analysis=json.loads((m/'evidence/native-natural-eos-failure-reanalysis.json').read_bytes());assert analysis['status']=='native_natural_eos_failure_and_complete_screen_replayed' and analysis['archive_sha256']==parents[failure]['archive_sha256'] and analysis['actual_generated_tokens']==74 and analysis['terminal_token_id']==248046
 files[analysis['early_eos_events_file']]=analysis['early_eos_events_sha256']
 for name in parents:files['results/'+name+'-verified.json']=parents[name]['proof_sha256']
 files['results/study-ssd-engine-cache/receipt.json']=hashlib.sha256(zipfile.ZipFile(args.proofs/(model+'-evidence.zip')).read('results/study-ssd-engine-cache/receipt.json')).hexdigest()
 files.update({'Garnet/test2026/cuda/qwen38_5090/'+name:sha((m/name).read_bytes()) for name in METHODS})
 for name,digest in files.items():
  if name.startswith('Garnet/'):assert sha((repo/name.removeprefix('Garnet/')).read_bytes())==digest,('Local source drift',name)
 data={'status':'fixed_length_eos_prepared_unrun','files':files,'runtime_dependencies':runtime,'parents':parents,'scope':'One correctness-only B1 ordinary thenMTP test on actual failed4Kimage input: natural74, forced128, natural74again and forced128SSE per mode. Prefix74and full paired IDs, source/runtime,43CPUcontrols, owned cache cleanup and immutable failure archive. Same1a1dDLL; no model rebuild,805quality repetition, timing or2x admission.'};target=m/'evidence/fixed-length-eos-preparation.json';raw=json.dumps(data,indent=2).encode();assert not target.exists() or target.read_bytes()==raw
 if not target.exists():target.write_bytes(raw)
 rows=[{'local':str(m/name),'target':'Garnet/test2026/cuda/qwen38_5090/'+name,'sha256':sha((m/name).read_bytes())} for name in METHODS]+[{'local':str(target),'target':'Garnet/test2026/cuda/qwen38_5090/evidence/'+target.name,'sha256':sha(target.read_bytes())}];raw=json.dumps({'uploads':rows,'scope':data['scope']},indent=2).encode();assert not args.plan.exists() or args.plan.read_bytes()==raw
 if not args.plan.exists():args.plan.write_bytes(raw)
 print(json.dumps({'status':data['status'],'preparation_sha256':sha(target.read_bytes()),'targets':len(rows)}));return 0
if __name__=='__main__':raise SystemExit(main())
