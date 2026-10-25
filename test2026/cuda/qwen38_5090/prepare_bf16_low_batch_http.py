"""Prepare the paired HTTP study only from the actual closed native model proof."""
from pathlib import Path
import argparse,hashlib,json,zipfile
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
METHODS=('native_bf16_low_batch_stream_worker.py','serve_native_bf16_low_batch_worker.bat','probe_bf16_low_batch_http.py','run_bf16_low_batch_http.py','prepare_bf16_low_batch_http.py','verify_bf16_low_batch_http.py','install_mmmu_quality_corpus.py','native_http_bridge.py','native_worker_io.py','native_prompt.py','benchmark_stream.py','benchmark_fixed_context_native.py','compare_http_runs.py','probe_reference_http.py','metrics.py','sample_vram.py','monitor_native_probe.py','run_logged.py','mmmu_http_quality.py','mmmu_quality_protocol.py','verify_mmmu_http_quality.py','test_bf16_low_batch_worker_contract.py','test_fixed_context_native_stream.py','test_mmmu_http_quality.py','test_mmmu_quality_protocol.py','evidence/fixed-context-4096-corpus.zip')
def main():
 a=argparse.ArgumentParser(description=__doc__);a.add_argument('--proofs',required=True,type=Path);a.add_argument('--corpus-archive',required=True,type=Path);a.add_argument('--plan',required=True,type=Path);args=a.parse_args()
 m=Path(__file__).resolve().parent;repo=m.parents[2];name='bf16-low-batch-model-consistency-attempt2'
 required=[args.proofs/(name+s) for s in ('-verified.json','-remote-verified.json','-evidence.zip')];missing=[str(p) for p in required if not p.is_file()]
 if missing:print(json.dumps({'status':'b1_http_model_prerequisite_not_ready_no_preparation','missing':missing}));return 2
 raw=(args.proofs/(name+'-remote-verified.json')).read_bytes();q=json.loads(raw);assert q==json.loads((args.proofs/(name+'-verified.json')).read_bytes())
 assert q['status']=='bf16_low_batch_model_consistency_replayed' and not q['performance_admitted'] and sha(args.proofs/(name+'-evidence.zip'))==q['archive_sha256']
 assert q['replay_verifier_sha256']==q['archived_replay_verifier_sha256']==sha(m/'verify_bf16_low_batch_model_fixed.py')
 assert [(v['label'],v['cases']) for v in q['variants']]==[('b4-smoke',4),('b1',8)] and q['variants'][1]['prefix_counts']==[0,1,2,4]
 with zipfile.ZipFile(args.proofs/(name+'-evidence.zip')) as z:
  closed=json.loads(z.read('results/'+name+'/workflow.json'));assert closed['exit_code']==0 and closed['cleanup_finished'] and closed['library_unchanged']
  assert closed['source_sha256']==closed['source_after_sha256'] and closed['runtime_dependencies']==closed['runtime_dependencies_after'] and closed['library_sha256']==q['library_sha256']
  files=dict(closed['source_sha256']);runtime=closed['runtime_dependencies'];receipt='results/study-ssd-engine-cache/receipt.json';files[receipt]=hashlib.sha256(z.read(receipt)).hexdigest()
  for n in ('b1/report.json','b4-smoke/report.json'):files['results/'+name+'/'+n]=hashlib.sha256(z.read('results/'+name+'/'+n)).hexdigest()
 files.update({'Garnet/test2026/cuda/qwen38_5090/'+n:sha(m/n) for n in METHODS})
 for n,h in files.items():
  if n.startswith('Garnet/'):assert sha(repo/n.removeprefix('Garnet/'))==h,('Local source/method drift',n)
 from install_mmmu_quality_corpus import ARCHIVE_SHA256,MANIFEST_SHA256
 assert sha(args.corpus_archive)==ARCHIVE_SHA256
 with zipfile.ZipFile(args.corpus_archive) as z:assert hashlib.sha256(z.read('manifest.json')).hexdigest()==MANIFEST_SHA256
 corpus_target='results/mmmu-quality-corpus.zip';files[corpus_target]=ARCHIVE_SHA256
 p={'status':'b1_http_study_prepared_unrun','files':files,'runtime_dependencies':runtime,'parent_proof_sha256':hashlib.sha256(raw).hexdigest(),'parent_archive_sha256':q['archive_sha256'],'corpus_archive':corpus_target,'corpus_archive_sha256':ARCHIVE_SHA256,'corpus_manifest_sha256':MANIFEST_SHA256,'scope':'One exclusive native B1 single-slot ordinary then MTP HTTP study, preserving queued concurrency.29CPU controls, one fixed805corpus installation/reuse,12protocol controls and805raw screening replies plus2frozen4Kboundary controls per mode; MTP must match ordinary actual IDs across screens/boundaries and all124full4Krequests. Complete fixed3968input/128output C1/2/4/8cohorts:64measured/60warmup requests per mode. Optional exploratory short C1/2/4/8pilot has8requests/2warmups/128output,seed3,identical JPEG nonces. Same-DLL/source/runtime and owned scratch retirement/immutable failures. No broad quality, matched/tuned references, INT4/4B completion or2x admission.'}
 target=m/'evidence/bf16-low-batch-http-preparation.json';data=json.dumps(p,indent=2).encode()
 if target.exists():assert target.read_bytes()==data
 else:target.write_bytes(data)
 rows=[{'local':str(m/n),'target':'Garnet/test2026/cuda/qwen38_5090/'+n,'sha256':sha(m/n)} for n in METHODS]+[{'local':str(args.corpus_archive),'target':corpus_target,'sha256':ARCHIVE_SHA256},{'local':str(target),'target':'Garnet/test2026/cuda/qwen38_5090/evidence/'+target.name,'sha256':sha(target)}]
 data=json.dumps({'uploads':rows,'scope':p['scope']},indent=2).encode()
 if args.plan.exists():assert args.plan.read_bytes()==data
 else:args.plan.write_bytes(data)
 print(json.dumps({'status':p['status'],'methods':len(METHODS),'preparation_sha256':sha(target),'parent_model_status':q['status']}));return 0
if __name__=='__main__':raise SystemExit(main())
