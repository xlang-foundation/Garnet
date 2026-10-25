"""Actual native3968+128 streaming cohorts from the immutable CPU-prepared corpus."""
from pathlib import Path
import argparse,concurrent.futures,datetime,hashlib,json,sys,uuid,zipfile
sys.path.insert(0,str(Path(__file__).resolve().parent))
from benchmark_stream import request
from metrics import summarize
from sample_vram import Sampler
ARCHIVE_SHA256='596991dd0907ccb8ae8ffb090737f59ba8da3faac181bb18fadc74ae0e4ef3ce'
MANIFEST_SHA256='25808fff53f326b6a1a970bb389173b2a54c1d1189248d40122fcdb959cb6987'
sha=lambda raw:hashlib.sha256(raw).hexdigest()
def final_ids(events,expected_prompt):
 values=[json.loads(e['data']) for e in events if e.get('data')!='[DONE]'];final=[v for v in values if 'response_token_ids' in v]
 if len(final)!=1:raise ValueError('One native final prompt/completion-ID record required')
 value=final[0];completion=value['response_token_ids'];tokens=[v['token_id'] for v in values if 'token_id' in v]
 prompt=value.get('prompt_token_ids')
 if len(expected_prompt)!=3968 or not isinstance(prompt,list) or any(type(n) is not int or not 0<=n<248320 for n in prompt) or prompt!=expected_prompt:raise ValueError('Actual4Kprompt IDs differ')
 if not isinstance(completion,list) or len(completion)!=128 or any(type(n) is not int or not 0<=n<248320 for n in completion) or tokens!=completion:raise ValueError('Actual128stream token IDs missing/different')
 if value.get('usage')!={'prompt_tokens':3968,'completion_tokens':128,'total_tokens':4096}:raise ValueError('Actual4Kusage differs')
 choices=value.get('choices')
 if not isinstance(choices,list) or len(choices)!=1 or type(choices[0].get('index')) is not int or choices[0]['index']!=0 or choices[0].get('finish_reason')!='length':raise ValueError('Fixed-length stream must finish at length, with one choice')
 return value
def main():
 a=argparse.ArgumentParser(description=__doc__);a.add_argument('--url',required=True);a.add_argument('--corpus-archive',type=Path,required=True);a.add_argument('--output',type=Path,required=True);a.add_argument('--metadata',type=Path,required=True);a.add_argument('--ordinary-benchmark',type=Path);args=a.parse_args()
 assert args.url=='http://127.0.0.1:18082' and not args.output.exists();assert sha(args.corpus_archive.read_bytes())==ARCHIVE_SHA256
 with zipfile.ZipFile(args.corpus_archive) as z:
  assert z.testzip() is None and len(z.namelist())==len(set(z.namelist()));raw=z.read('manifest.json');assert sha(raw)==MANIFEST_SHA256;corpus=json.loads(raw);assert len(corpus['rows'])==124 and len(corpus['cohorts'])==8
  rows={r['key']:r for r in corpus['rows']};prepared={}
  for key,row in rows.items():
   wire=z.read(row['request_file']);ids_raw=z.read(row['expected_prompt_ids_file']);assert sha(wire)==row['request_sha256'] and sha(ids_raw)==row['expected_prompt_ids_sha256'];body=json.loads(wire);ids=json.loads(ids_raw)
   assert body['stream'] and body['max_tokens']==128 and body.get('ignore_eos') is True and len(ids)==3968 and row['total_context_capacity']==4096
   prepared[key]=(body,ids)
 ordinary_ids={};ordinary_manifest_sha=None
 if args.ordinary_benchmark is not None:
  manifests=list(args.ordinary_benchmark.glob('*-manifest.json'));assert len(manifests)==1;raw=manifests[0].read_bytes();ordinary=json.loads(raw);metadata=json.loads(args.metadata.read_bytes())
  assert ordinary['status']=='completed' and ordinary['corpus_archive_sha256']==ARCHIVE_SHA256 and ordinary['engine_metadata']['generation_mode']=='ordinary' and ordinary['engine_metadata']['library_sha256']==metadata['library_sha256'] and ordinary['engine_metadata']['model_proof_sha256']==metadata['model_proof_sha256']
  ordinary_manifest_sha=sha(raw)
  for key,(_,ids) in prepared.items():
   events=[json.loads(line) for line in (args.ordinary_benchmark/(ordinary['run_id']+'-'+key+'-events.jsonl')).read_text().splitlines()];ordinary_ids[key]=final_ids(events,ids)['response_token_ids']
 args.output.mkdir();out=args.output;run=uuid.uuid4().hex;prefix=run+'-';cohorts=[];sampler=None
 manifest={'run_id':run,'status':'running','engine':'garnet','utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'argv':sys.argv,'corpus_archive_sha256':ARCHIVE_SHA256,'corpus_manifest_sha256':MANIFEST_SHA256,'context_capacity':4096,'actual_input_tokens_required':3968,'actual_completion_tokens_required':128,'engine_metadata':json.loads(args.metadata.read_bytes()),'engine_metadata_sha256':sha(args.metadata.read_bytes()),'cohorts':cohorts,'ordinary_manifest_sha256':ordinary_manifest_sha,'performance_admitted':False,'scope':'Actual fixed-input3968/output128 native loopback streaming cohorts, including all template/image tokens. Artificial repeated-note corpus; no broad-quality, reference-equivalence or2x promotion.'}
 def save():(out/(prefix+'manifest.json')).write_text(json.dumps(manifest,indent=2),encoding='utf8')
 def send(key):
  body,ids=prepared[key];stem=prefix+key;result=request(args.url,body,out,stem,128)
  if result['status']!='completed':raise RuntimeError('Preserved failed4Krequest: '+key)
  events=[json.loads(line) for line in (out/(stem+'-events.jsonl')).read_text().splitlines()];actual=final_ids(events,ids)
  if ordinary_ids and actual['response_token_ids']!=ordinary_ids[key]:raise ValueError('MTP4K IDs differ from actual ordinary run: '+key)
  assert result['prompt_tokens']==3968 and result['generated_tokens']==128;return result
 save()
 try:
  sampler=Sampler(out/(prefix+'vram.jsonl'));manifest['gpu']=sampler.device;sampler.start()
  for cohort in corpus['cohorts']:
   case=cohort['case'];concurrency=cohort['concurrency'];sampler.phase='warmup-'+case+'-c'+str(concurrency)
   for wave in cohort['warmups']:
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
     for future in [pool.submit(send,key) for key in wave]:future.result()
   phase=case+'-c'+str(concurrency);sampler.phase=phase;sampler.sample()
   with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:measured=[f.result() for f in [pool.submit(send,key) for key in cohort['measured']]]
   sampler.sample();summary=summarize(measured);summary.update(sampler.peak(phase));summary.update(case=case,concurrency=concurrency,request_keys=[r['key'] for r in measured],single_token_arrivals=all(r['single_token_arrivals'] for r in measured),prompt_tokens=[r['prompt_tokens'] for r in measured]);cohorts.append(summary);save();print(json.dumps(summary),flush=True)
  manifest['status']='completed'
 except BaseException as error:manifest.update(status='failed',error=repr(error));raise
 finally:
  try:
   if sampler is not None:sampler.close()
  except BaseException as error:manifest.update(status='failed',error=repr(error));raise
  finally:manifest['sampling_errors']=sampler.errors if sampler is not None else [];save()
 return 0
if __name__=='__main__':raise SystemExit(main())
