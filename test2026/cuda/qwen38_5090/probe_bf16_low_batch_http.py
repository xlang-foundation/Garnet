"""Prepared actual B1 HTTP controls,805screen,4K controls and exploratory pilot.

An outer controller must establish terminal model gates, exclusive ownership,
source/runtime identity and scratch cleanup. This probe cannot replace it.
"""
from pathlib import Path
import argparse,concurrent.futures,copy,hashlib,json,subprocess,sys,threading,time,zipfile
from urllib.request import Request,urlopen
from urllib.error import HTTPError
from native_http_bridge import Worker,Handler,ThreadingHTTPServer
from benchmark_stream import request as stream_request

def main():
 a=argparse.ArgumentParser(description=__doc__);a.add_argument('--root',type=Path,required=True);a.add_argument('--output',type=Path,required=True);a.add_argument('--corpus',type=Path,required=True);a.add_argument('--mode',choices=('ordinary','mtp'),required=True);a.add_argument('--ordinary-controls',type=Path);a.add_argument('--exploratory-pilot',action='store_true');args=a.parse_args()
 root=args.root;out=args.output;m=root/'Garnet/test2026/cuda/qwen38_5090';sys.path.insert(0,str(root/'Garnet'))
 from test2026.cuda.qwen38_5090.mmmu_http_quality import load_corpus,collect
 from test2026.cuda.qwen38_5090.verify_mmmu_http_quality import replay
 sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();model='bf16-low-batch-model-consistency-attempt2';proof_path=root/'results'/(model+'-verified.json');proof=json.loads(proof_path.read_bytes());library=sha(root/'build/bin/garnet.dll')
 assert proof['status']=='bf16_low_batch_model_consistency_replayed' and proof['library_sha256']==library and sha(root/'results'/(model+'-evidence.zip'))==proof['archive_sha256']
 native=json.loads((root/'results'/model/'b1/report.json').read_bytes());assert native['status']=='bf16_low_batch_model_consistency_finished'
 _,corpus_rows,gold=load_corpus(args.corpus)
 if args.mode=='mtp':
  assert args.ordinary_controls is not None,'MTP requires actual completed ordinary controls'
  ordinary_report=json.loads((args.ordinary_controls/'report.json').read_bytes());assert ordinary_report['status']=='bf16_low_batch_http_profile_collected' and ordinary_report['fixed_context_benchmark_collected'] and ordinary_report['mode']=='ordinary' and ordinary_report['library_sha256']==library and ordinary_report['model_proof_sha256']==sha(proof_path)
  assert replay(args.corpus,args.ordinary_controls/'mmmu-quality')['status']=='adapted_mmmu_http_quality_responses_replayed'
 assert not out.exists();out.mkdir()
 cache=json.loads((root/'results/study-ssd-engine-cache/receipt.json').read_bytes())['cache_root']
 worker=Worker(['cmd.exe','/d','/c',str(m/'serve_native_bf16_low_batch_worker.bat'),args.mode,cache],out/'worker');server=thread=None;url='http://127.0.0.1:18082'
 report={'status':'running','mode':args.mode,'physical_batch':1,'queued_concurrency':True,'model_proof_sha256':sha(proof_path),'library_sha256':library,'protocol_cases':[],'boundary_controls':[],'broad_quality_qualified':False,'performance_admitted':False,'scope':'Actual single-slot serving controls, adapted805 screening and two frozen3968-input/128-output-reserve controls precede complete124request fixed4Kcohorts. MTP4Kmust match actual ordinary IDs. Optional short-prompt pilot and all timings remain exploratory; no broad-quality/reference-equivalence/2x admission.'}
 def save():(out/'report.json').write_text(json.dumps(report,indent=2),encoding='utf8')
 save()
 def expected(key):
  cases=[c for c in native['cases'] if c['key']==key];assert len(cases)==2 and cases[0]['generated_ids']==cases[1]['generated_ids'];return cases[0]['generated_ids'][0]
 def nonstream(key,body,prompt_ids,completion=None):
  body=copy.deepcopy(body);body['stream']=False;wire=json.dumps(body,separators=(',',':')).encode();(out/(key+'-request.json')).write_bytes(wire)
  try:
   with urlopen(Request(url+'/v1/chat/completions',data=wire,headers={'Content-Type':'application/json'}),timeout=300) as response:raw=response.read()
  except HTTPError as error:(out/(key+'-error.bin')).write_bytes(error.read());raise
  (out/(key+'-response.json')).write_bytes(raw);value=json.loads(raw);choice=value['choices'][0];ids=choice['response_token_ids']
  assert choice['prompt_token_ids']==prompt_ids and ids and len(prompt_ids)+len(ids)<=4096 and len(ids)<=body['max_tokens']
  assert value['usage']=={'prompt_tokens':len(prompt_ids),'completion_tokens':len(ids),'total_tokens':len(prompt_ids)+len(ids)}
  if completion is not None:assert ids==completion
  return {'key':key,'prompt_ids_exact':True,'completion_ids_exact':completion is not None,'completion_tokens':len(ids),'finish_reason':choice['finish_reason']}
 try:
  while not worker.healthy():
   assert worker.process.poll() is None and worker.failure is None,worker.failure;time.sleep(.25)
  ready=worker.ready_info;assert ready['batch']==1 and ready['capacity']==4096 and ready['prefill_chunk']==32 and ready['chunk']==4 and ready['library_sha256']==library and ready['generation_mode']==args.mode and ready['bf16_kv_low_batch_state']
  report['ready']=ready;save();server=ThreadingHTTPServer(('127.0.0.1',18082),Handler);server.worker=worker;server.model='qwen38-27b-nvfp4';thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
  names=['s3-text-c1-r0-greedy','s3-image-c1-r0-greedy','red','green'];oracle=root/'results/native-mtp3-http'
  fixtures={key:(json.loads((oracle/(key+'-request.json')).read_bytes()),json.loads((oracle/(key+'-response.json')).read_bytes())['choices'][0]['prompt_token_ids']) for key in names}
  for repeat in range(2):
   with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
    futures=[pool.submit(nonstream,'protocol-'+str(repeat)+'-'+key,fixtures[key][0],fixtures[key][1],expected(key)) for key in names];report['protocol_cases'].extend(f.result() for f in futures)
   save()
  for key in names:
   body=copy.deepcopy(fixtures[key][0]);body['stream']=True;stem='stream-'+key;result=stream_request(url,body,out,stem,body['max_tokens']);assert result['status']=='completed'
   events=[json.loads(line) for line in (out/(stem+'-events.jsonl')).read_text().splitlines()];values=[json.loads(e['data']) for e in events if e.get('data')!='[DONE]'];final=[v for v in values if 'response_token_ids' in v];assert len(final)==1
   assert final[0]['prompt_token_ids']==fixtures[key][1] and final[0]['response_token_ids']==expected(key) and result['generated_tokens']==len(expected(key))
   report['protocol_cases'].append({'key':stem,'actual_stream_ids_exact':True,'actual_stream_tokens':result['generated_tokens']});save()
  identity={'engine':'native','model_alias':'qwen38-27b-nvfp4','library_sha256':library,'model_proof_sha256':sha(proof_path),'generation_mode':args.mode,'physical_batch':1,'ready':ready,'worker_sha256':sha(m/'native_bf16_low_batch_stream_worker.py'),'scope':'Declared by this actual loopback server launcher; client independently reports identity as unattested.'}
  (out/'server-identity.json').write_text(json.dumps(identity,indent=2),encoding='utf8');collection=collect(args.corpus,corpus_rows,gold,out/'mmmu-quality','native','qwen38-27b-nvfp4',url,identity,timeout=300);assert collection['status']=='completed_needs_independent_replay'
  screening=replay(args.corpus,out/'mmmu-quality');assert screening['status']=='adapted_mmmu_http_quality_responses_replayed';(out/'mmmu-quality-replayed.json').write_text(json.dumps(screening,indent=2),encoding='utf8');report['adapted_screening']=screening;save()
  if args.mode=='mtp':
   mismatches=[]
   for row in corpus_rows:
    relative=Path('mmmu-quality/cases')/row['id']/'response.bin';ordinary=json.loads((args.ordinary_controls/relative).read_bytes())['choices'][0];current=json.loads((out/relative).read_bytes())['choices'][0]
    if ordinary['prompt_token_ids']!=current['prompt_token_ids'] or ordinary['response_token_ids']!=current['response_token_ids']:mismatches.append(row['id'])
   report['ordinary_mtp_screening_token_mismatches']=mismatches;save();assert not mismatches,'MTP differs from actual ordinary screening IDs'
  with zipfile.ZipFile(m/'evidence/fixed-context-4096-corpus.zip') as z:
   assert hashlib.sha256(z.read('manifest.json')).hexdigest()=='829d7770f2d749e30a66e027940224e8ab77a488d0b5f9c456e4809ae845987c';manifest=json.loads(z.read('manifest.json'))
   for case in ('text','image'):
    key='ctx4096-s3-'+case+'-c1-warmup-0-0';row=next(r for r in manifest['rows'] if r['key']==key);wire=z.read(row['request_file']);ids_raw=z.read(row['expected_prompt_ids_file']);assert hashlib.sha256(wire).hexdigest()==row['request_sha256'] and hashlib.sha256(ids_raw).hexdigest()==row['expected_prompt_ids_sha256']
    ids=json.loads(ids_raw);assert len(ids)==3968 and row['completion_reserved_tokens']==128;completion=None
    if args.mode=='mtp':
     ordinary=json.loads((args.ordinary_controls/('boundary-'+case+'-response.json')).read_bytes());assert ordinary['choices'][0]['prompt_token_ids']==ids;completion=ordinary['choices'][0]['response_token_ids'];assert len(completion)==128
    result=nonstream('boundary-'+case,json.loads(wire),ids,completion);assert result['completion_tokens']==128
    report['boundary_controls'].append(result);save()
  report['status']='protocol_screening_and_two_boundary_controls_passed';save()
  context_metadata=out/'fixed-context-metadata.json';context_metadata.write_text(json.dumps({'ready':ready,'physical_batch':1,'queued_concurrency':True,'generation_mode':args.mode,'library_sha256':library,'model_proof_sha256':sha(proof_path),'protocol_report_sha256_before_timing':sha(out/'report.json'),'broad_quality_qualified':False,'performance_admitted':False,'scope':'Actual fixed3968input/128output native cohorts; ordinary/MTP paired IDs, frozen corpus, no matched-reference or2x admission.'},indent=2),encoding='utf8')
  context_command=[sys.executable,str(m/'benchmark_fixed_context_native.py'),'--url',url,'--corpus-archive',str(m/'evidence/fixed-context-4096-corpus.zip'),'--output',str(out/'benchmark4k'),'--metadata',str(context_metadata)]
  if args.mode=='mtp':context_command+=['--ordinary-benchmark',str(args.ordinary_controls/'benchmark4k')]
  subprocess.run(context_command,check=True);report['fixed_context_benchmark_collected']=True;save()
  if args.exploratory_pilot:
   metadata=out/'benchmark-metadata.json';metadata.write_text(json.dumps({'ready':ready,'physical_batch':1,'queued_concurrency':True,'generation_mode':args.mode,'library_sha256':library,'model_proof_sha256':sha(proof_path),'protocol_report_sha256_before_pilot':sha(out/'report.json'),'corpus_manifest_sha256':screening['corpus_manifest_sha256'],'boundary_controls':report['boundary_controls'],'broad_quality_qualified':False,'performance_admitted':False,'scope':'Exploratory short-prompt C1/2/4/8 pilot after actual B1 protocol/screen/boundary controls; not the full4Kbenchmark or2x qualification.'},indent=2),encoding='utf8')
   subprocess.run([sys.executable,str(m/'benchmark_stream.py'),'--url',url,'--engine','garnet','--output',str(out/'benchmark'),'--metadata',str(metadata),'--concurrency','1','2','4','8','--requests','8','--warmups','2','--max-tokens','128','--workload-seed','3','--image-format','jpeg','--image-order','image-first','--image-nonce-pattern','binary-grid'],check=True)
   report['exploratory_short_pilot_collected']=True;save()
  report['status']='bf16_low_batch_http_profile_collected';save()
 except BaseException as error:report.update(status='failed',error=repr(error));save();raise
 finally:
  if server is not None:server.shutdown();server.server_close()
  if thread is not None:thread.join()
  worker.close()
 return 0
if __name__=='__main__':raise SystemExit(main())
