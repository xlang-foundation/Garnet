"""Replay B8 internal row/state/4K admission; no external quality or speed claim."""
from pathlib import Path
import argparse,base64,hashlib,io,json,struct,sys,zipfile
sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
from test2026.cuda.qwen38_5090.bf16_batch8_admission import ORDERS,compare_permuted,compare_vector_rows
sha=lambda raw:hashlib.sha256(raw).hexdigest()
METHODS='Garnet/test2026/cuda/qwen38_5090/'

def replay_cases(report,read,groups):
 assert report['physical_batch']==8 and report['capacity']==4096 and report['status']=='bf16_batch8_ordinary_admission_collected_needs_replay'
 assert all(report[key] is False for key in ('mtp_executed','http_executed','quality_qualified','performance_admitted'))
 def state(value):
  logical=value['logical'];assert compare_permuted(logical,logical,ORDERS[0])
  assert set(value['vectors'])=={'hidden','normalized'};raws={}
  for kind,entry in value['vectors'].items():
   assert Path(entry['file']).name==entry['file'];raw=read(entry['file'])
   assert len(raw)==entry['bytes']==8*5120*4 and sha(raw)==entry['sha256']
   assert all(__import__('math').isfinite(v[0]) for v in struct.iter_unpack('<f',raw))
   raws[kind]=raw
  return logical,raws
 def same(left,right,order):
  assert compare_permuted(left[0],right[0],order)
  assert all(compare_vector_rows(left[1][kind],right[1][kind],order) for kind in ('hidden','normalized'))
 empty=state(report['empty_state']);assert empty[0]['slots']==[0]*8;same(empty,state(report['final_reset']),ORDERS[0])
 assert len(report['cases'])==9;references={};summaries=[]
 for index,case in enumerate(report['cases']):
  label,inputs,limit=groups[index//3];order=ORDERS[index%3]
  assert case['group']==label and case['repeat']==index%3 and case['order']==list(order) and case['actual_limit']==limit and case['inverse_permutation_exact'] is True and case['performance_admitted'] is False
  assert len(case['inputs'])==len(case['generated_ids'])==len(case['text'])==8
  for physical,original in enumerate(order):
   assert case['inputs'][physical]==inputs[original]
   ids=case['generated_ids'][physical];assert len(ids)==limit and all(type(v) is int and 0<=v<248320 for v in ids)
   assert isinstance(case['text'][physical],str)
  reset=state(case['reset_state']);same(empty,reset,ORDERS[0]);prefill=state(case['prefill_state']);final=state(case['final_state'])
  lengths=[len(inputs[i]['prompt_ids']) for i in order]
  assert prefill[0]['slots']==lengths and final[0]['slots']==[n+limit-1 for n in lengths]
  if label not in references:references[label]=(case,prefill,final)
  previous,previous_prefill,previous_final=references[label]
  assert all(case['generated_ids'][row]==previous['generated_ids'][original] for row,original in enumerate(order))
  same(previous_prefill,prefill,order);same(previous_final,final,order)
  summaries.append({'group':label,'repeat':index%3,'physical_rows':8,'actual_generated_ids_each':limit,'all_input_ids_bound':True,'complete_state_and_raw_vectors_inverse_permutation_exact':True})
 pool=report['pool_before'];assert pool['embedding_mode']=='host_mapped_bf16' and pool['embedding_table_bytes']==0 and pool['host_embedding_mapped_tensor_bytes']==2542796800
 assert (pool['compressed_projection_pool_entries'],pool['compressed_nvfp4_entries'],pool['compressed_fp8_entries'])==(336,128,208)
 assert pool['head_precision']=='modelopt_nvfp4_w4a4' and pool['batch_top1_parallel_enabled']
 assert report['commit_helper']['ready'] and report['prefill_commit_helper']['ready']
 return summaries

def main():
 a=argparse.ArgumentParser(description=__doc__);a.add_argument('--proofs',type=Path,required=True);args=a.parse_args();name='bf16-batch8-model-admission-attempt1';archive=args.proofs/(name+'-evidence.zip');base='results/'+name+'/'
 with zipfile.ZipFile(archive) as z:
  doc=lambda n:json.loads(z.read(n));manifest=doc('evidence-manifest.json');assert z.testzip() is None and len(z.namelist())==len(set(z.namelist()))==len(manifest['files'])+1
  for relative,digest in manifest['files'].items():assert sha(z.read(relative))==digest
  w=doc(base+'workflow.json');praw=z.read(METHODS+'evidence/bf16-batch8-admission-run-preparation.json');p=json.loads(praw)
  assert sha(praw)==w['preparation_sha256'] and p['status']=='bf16_batch8_model_admission_prepared_gpu_unrun'
  assert w['source_sha256']==w['source_after_sha256']==p['files'] and w['runtime_dependencies']==w['runtime_dependencies_after']==p['runtime_dependencies']
  assert w['library_unchanged'] and w['cleanup_finished'] and w['library_sha256']==p['runtime_dependencies']['garnet.dll'] and w['parents']==p['parents']
  for relative,digest in p['files'].items():assert sha(z.read(relative))==digest
  for helper in ('bf16_batch8_admission.py','bf16_packed_state_gate.py'):
   assert sha(Path(__file__).with_name(helper).read_bytes())==sha(z.read(METHODS+helper))
  for parent,row in p['parents'].items():
   raw=z.read('results/'+parent+'-verified.json');proof=json.loads(raw);assert sha(raw)==row['proof_sha256'] and proof['status']==row['status'] and proof['archive_sha256']==row['archive_sha256']==sha((args.proofs/(parent+'-evidence.zip')).read_bytes())
  reference=w['closed_reference'];assert reference['cleanup_finished'] is True
  with zipfile.ZipFile(args.proofs/(reference['job']+'-evidence.zip')) as previous:
   assert sha((args.proofs/(reference['job']+'-evidence.zip')).read_bytes())==reference['archive_sha256']
   closed=json.loads(previous.read('results/'+reference['job']+'/workflow.json'));assert closed==doc('results/'+reference['job']+'/workflow.json') and closed['exit_code']==reference['exit_code'] and closed['cleanup_finished'] and type(closed['server_exit_code']) is int
  passed=w['exit_code']==0;summaries=[];peak=None
  if passed:
   assert w['status']=='bf16_batch8_model_admission_collected_needs_replay' and b'Ran 28 tests' in z.read(base+'cpu-ledger/commands.log') and b'\nOK' in z.read(base+'cpu-ledger/commands.log')
   for ledger in ('cpu-ledger','gpu-ledger'):
    rows=[json.loads(line) for line in z.read(base+ledger+'/commands.jsonl').splitlines()];assert len(rows)==2 and rows[-1]['event']=='finished' and rows[-1]['exit_code']==0
   short=[]
   for key in ('s3-text-c1-r0-greedy','s3-image-c1-r0-greedy','red','green'):
    raw=z.read('results/native-mtp3-http/'+key+'-request.json');response=doc('results/native-mtp3-http/'+key+'-response.json');body=json.loads(raw);content=body['messages'][0]['content'];image_hash=None
    if isinstance(content,list):image_hash=sha(base64.b64decode(content[0]['image_url']['url'].split(',',1)[1],validate=True))
    short.append({'key':key,'request_sha256':sha(raw),'prompt_ids':response['choices'][0]['prompt_token_ids'],'image_sha256':image_hash})
   groups=[('short-mixed',short+short,8)]
   corpus_raw=z.read(METHODS+'evidence/fixed-length-4096-corpus.zip');assert sha(corpus_raw)=='596991dd0907ccb8ae8ffb090737f59ba8da3faac181bb18fadc74ae0e4ef3ce'
   with zipfile.ZipFile(io.BytesIO(corpus_raw)) as corpus:
    raw=corpus.read('manifest.json');assert sha(raw)=='25808fff53f326b6a1a970bb389173b2a54c1d1189248d40122fcdb959cb6987';frozen=json.loads(raw);rows={v['key']:v for v in frozen['rows']}
    for case in ('text','image'):
     cohort=next(c for c in frozen['cohorts'] if c['case']==case and c['concurrency']==8);inputs=[]
     for key in cohort['measured']:
      row=rows[key];raw=corpus.read(row['request_file']);ids=corpus.read(row['expected_prompt_ids_file']);assert sha(raw)==row['request_sha256'] and sha(ids)==row['expected_prompt_ids_sha256'];body=json.loads(raw);assert body['ignore_eos'] is True
      content=body['messages'][0]['content'];image_hash=sha(base64.b64decode(content[0]['image_url']['url'].split(',',1)[1],validate=True)) if isinstance(content,list) else None
      inputs.append({'key':key,'request_sha256':sha(raw),'prompt_ids':json.loads(ids),'image_sha256':image_hash})
     assert len(inputs)==8 and all(len(v['prompt_ids'])==3968 for v in inputs);groups.append(('full4k-'+case,inputs,128))
   for label,inputs,limit in groups:
    for i,row in enumerate(inputs):
     if row['image_sha256']:assert sha(z.read(base+'model/'+label+'-row'+str(i)+'.jpg'))==row['image_sha256']
   report=doc(base+'model/report.json');assert report['library_sha256']==w['library_sha256'];summaries=replay_cases(report,lambda n:z.read(base+'model/'+n),groups)
   monitor=doc(base+'resources/resource-monitor.json');samples=[json.loads(line) for line in z.read(base+'resources/resource-vram.jsonl').splitlines()];peak=max(row['used_bytes'] for row in samples)
   assert monitor['child_exit_code']==0 and monitor['gpu']['uuid']=='GPU-265ee66c-9b79-cd3b-0c63-dd6942ddb5e6' and monitor['sampled_memory']['sampled_peak_adapter_used_bytes']==peak
  result={'status':'bf16_batch8_internal_model_admission_replayed' if passed else 'bf16_batch8_model_admission_failure_preserved','archive_sha256':sha(archive.read_bytes()),'files_verified':len(manifest['files']),'library_sha256':w['library_sha256'],'cases':summaries,'correctness_sampled_peak_adapter_used_bytes':peak,'replay_verifier_sha256':sha(Path(__file__).read_bytes()),'archived_replay_verifier_sha256':sha(z.read(METHODS+'verify_bf16_batch8_admission.py')),'external_b1_b4_numerical_equivalence_proven':False,'broad_quality_qualified':False,'mtp_qualified':False,'http_batching_qualified':False,'performance_admitted':False,'two_times_target_achieved':False,'scope':'Actual full eight-row internal permutation/reset, complete logical state hashes and finite raw hidden/normalized byte replay, short and3968+128 text/image execution/memory only. Not independent B1/B4 model-math equivalence,805quality, MTP, serving/throughput or2x admission.'}
 target=args.proofs/(name+'-verified.json');raw=json.dumps(result,indent=2).encode();assert not target.exists() or target.read_bytes()==raw
 if not target.exists():target.write_bytes(raw)
 print(json.dumps(result,indent=2));return 0 if passed else 1
if __name__=='__main__':raise SystemExit(main())
