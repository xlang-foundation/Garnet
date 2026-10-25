"""Prepare full timing only from both actual model and EOS-policy proof pairs."""
from pathlib import Path
import argparse,hashlib,json,zipfile
from prepare_tokenizer_nfc_http import METHODS as PREVIOUS_METHODS
sha=lambda raw:hashlib.sha256(raw).hexdigest()
RENAMES={'native_tokenizer_nfc_stream_worker.py':'native_fixed_length_stream_worker.py','serve_native_tokenizer_nfc_worker.bat':'serve_native_fixed_length_worker.bat','probe_tokenizer_nfc_http.py':'probe_fixed_length_http.py','run_tokenizer_nfc_http.py':'run_fixed_length_http.py','prepare_tokenizer_nfc_http.py':'prepare_fixed_length_http.py','verify_tokenizer_nfc_http.py':'verify_fixed_length_http.py','native_http_bridge.py':'native_fixed_length_http_bridge.py','benchmark_fixed_context_native.py':'benchmark_fixed_length_native.py','test_tokenizer_nfc_worker_contract.py':'test_fixed_length_worker_contract.py','test_fixed_context_native_stream.py':'test_fixed_length_native_stream.py','evidence/fixed-context-4096-corpus.zip':'evidence/fixed-length-4096-corpus.zip','evidence/tokenizer-nfc-http-method-derivation.json':'evidence/fixed-length-http-method-derivation.json'}
METHODS=tuple(RENAMES.get(n,n) for n in PREVIOUS_METHODS)+('fixed_length_policy.py','benchmark_fixed_length_reference.py','test_fixed_length_reference.py','vllm_cache_control.py','verify_fixed_length_eos.py')

def main():
 a=argparse.ArgumentParser(description=__doc__);a.add_argument('--proofs',required=True,type=Path);a.add_argument('--corpus-archive',required=True,type=Path);a.add_argument('--plan',required=True,type=Path);args=a.parse_args();m=Path(__file__).resolve().parent;repo=m.parents[2];model='tokenizer-nfc-model-consistency-attempt1';eos='fixed-length-eos-consistency-attempt1';parents={}
 missing=[str(args.proofs/(name+suffix)) for name in (model,eos) for suffix in ('-verified.json','-remote-verified.json','-evidence.zip') if not (args.proofs/(name+suffix)).is_file()]
 if missing:print(json.dumps({'status':'fixed_length_http_prerequisites_missing_no_preparation','missing':missing}));return 2
 for name,status in ((model,'tokenizer_nfc_model_consistency_replayed'),(eos,'fixed_length_eos_consistency_replayed')):
  raw=(args.proofs/(name+'-remote-verified.json')).read_bytes();q=json.loads(raw);assert q==json.loads((args.proofs/(name+'-verified.json')).read_bytes()) and q['status']==status and q['archive_sha256']==sha((args.proofs/(name+'-evidence.zip')).read_bytes());parents[name]=(raw,q)
 model_raw,q=parents[model];eos_raw,eq=parents[eos];assert q['library_sha256']==eq['library_sha256'] and eq['replay_verifier_sha256']==sha((m/'verify_fixed_length_eos.py').read_bytes()) and [(v['mode'],v['actual_requests']) for v in eq['profiles']]==[('ordinary',4),('mtp',4)] and eq['profiles'][1]['ordinary_mtp_exact']
 assert q['replay_verifier_sha256']==q['archived_replay_verifier_sha256']==sha((m/'verify_tokenizer_nfc_model.py').read_bytes()) and [(v['label'],v['cases']) for v in q['variants']]==[('b4-smoke',4),('b1',8)]
 with zipfile.ZipFile(args.proofs/(eos+'-evidence.zip')) as z:
  w=json.loads(z.read('results/'+eos+'/workflow.json'));assert w['cleanup_finished'] and w['exit_code']==0 and w['library_unchanged'] and w['source_sha256']==w['source_after_sha256'] and w['runtime_dependencies']==w['runtime_dependencies_after'];files=dict(w['source_sha256']);runtime=w['runtime_dependencies']
 with zipfile.ZipFile(args.proofs/(model+'-evidence.zip')) as z:
  for name in ('b1/report.json','b4-smoke/report.json'):files['results/'+model+'/'+name]=sha(z.read('results/'+model+'/'+name))
 files['results/'+model+'-verified.json']=sha(model_raw);files['results/'+eos+'-verified.json']=sha(eos_raw)
 derivation=json.loads((m/'evidence/fixed-length-http-method-derivation.json').read_bytes());archive=args.proofs/'tokenizer-nfc-http-study-attempt1-evidence.zip';assert sha(archive.read_bytes())==derivation['parent_source_archive_sha256']
 with zipfile.ZipFile(archive) as z:
  for target,row in derivation['files'].items():
   original=z.read(row['source']);assert sha(original)==row['source_sha256'];text=original.decode()
   for before,after,count in row['changes']:assert text.count(before)==count;text=text.replace(before,after)
   assert text.encode()==(m/target).read_bytes() and sha(text.encode())==row['result_sha256']
 files.update({'Garnet/test2026/cuda/qwen38_5090/'+name:sha((m/name).read_bytes()) for name in METHODS})
 for name,digest in files.items():
  if name.startswith('Garnet/'):assert sha((repo/name.removeprefix('Garnet/')).read_bytes())==digest,('Local source/method drift',name)
 from install_mmmu_quality_corpus import ARCHIVE_SHA256,MANIFEST_SHA256
 assert sha(args.corpus_archive.read_bytes())==ARCHIVE_SHA256
 with zipfile.ZipFile(args.corpus_archive) as z:assert sha(z.read('manifest.json'))==MANIFEST_SHA256
 files['results/mmmu-quality-corpus.zip']=ARCHIVE_SHA256
 p={'status':'fixed_length_http_prepared_unrun','files':files,'runtime_dependencies':runtime,'parent_proof_sha256':sha(model_raw),'parent_archive_sha256':q['archive_sha256'],'eos_proof_sha256':sha(eos_raw),'eos_archive_sha256':eq['archive_sha256'],'corpus_archive':'results/mmmu-quality-corpus.zip','corpus_archive_sha256':ARCHIVE_SHA256,'corpus_manifest_sha256':MANIFEST_SHA256,'scope':'One exclusive B1 ordinary thenMTP full HTTP study after qualified8requestEOS regression, unchanged1a1dDLL/model source.43CPU controls,12protocol/805natural-stop adapted screening/2fixed-length boundaries and124full4Krequests with explicit ignore_eos:true and exact3968input+128actual outputs per mode. All124MTP completions must match ordinary.124short exploratory natural-stop requests per mode remain separate. Raw events/usage/phase-specific sampled VRAM and all gaps retained, immutable failures and owned scratch cleanup. Matched strongest references/tuning/quality/INT4/4B/repetitions and2x remain open.'}
 target=m/'evidence/fixed-length-http-preparation.json';raw=json.dumps(p,indent=2).encode();assert not target.exists() or target.read_bytes()==raw
 if not target.exists():target.write_bytes(raw)
 rows=[{'local':str(m/name),'target':'Garnet/test2026/cuda/qwen38_5090/'+name,'sha256':sha((m/name).read_bytes())} for name in METHODS]+[{'local':str(args.corpus_archive),'target':p['corpus_archive'],'sha256':ARCHIVE_SHA256},{'local':str(target),'target':'Garnet/test2026/cuda/qwen38_5090/evidence/'+target.name,'sha256':sha(target.read_bytes())}];raw=json.dumps({'uploads':rows,'scope':p['scope']},indent=2).encode();assert not args.plan.exists() or args.plan.read_bytes()==raw
 if not args.plan.exists():args.plan.write_bytes(raw)
 print(json.dumps({'status':p['status'],'methods':len(METHODS),'preparation_sha256':sha(target.read_bytes())}));return 0
if __name__=='__main__':raise SystemExit(main())
