"""Frozen4K reference HTTP cohorts after actual nonstreaming ID controls.

SGLang0.5.22 rejects ID return flags on streaming chat. Timed requests therefore
retain the exact frozen bytes/semantic fields and use continuous usage, while
separate untimed requests attest full prompt/completion IDs and greedy text.
This records the limitation; it does not reconstruct streamed IDs from text.
"""
from pathlib import Path
import argparse,concurrent.futures,copy,datetime,hashlib,json,sys,uuid,zipfile
from urllib.request import Request,urlopen
from urllib.error import HTTPError
sys.path.insert(0,str(Path(__file__).resolve().parent))
from benchmark_fixed_context_native import ARCHIVE_SHA256,MANIFEST_SHA256
from benchmark_stream import request
from metrics import summarize
from sample_vram import Sampler
from vllm_cache_control import reset_idle_caches,verify_after_run
sha=lambda raw:hashlib.sha256(raw).hexdigest()

def control_request(body,engine):
 if engine not in ('vllm','sglang'):raise ValueError('Reference engine required')
 result=copy.deepcopy(body);result['stream']=False;result.pop('stream_options',None);result['return_token_ids']=True
 if engine=='sglang':result['return_prompt_token_ids']=True
 return result

def control_response(value,engine,expected_prompt):
 if engine not in ('vllm','sglang') or not isinstance(value,dict) or 'error' in value:raise ValueError('Reference response required')
 choices=value.get('choices')
 if not isinstance(choices,list) or len(choices)!=1:raise ValueError('One reference choice required')
 choice=choices[0]
 if type(choice.get('index')) is not int or choice['index']!=0 or choice.get('finish_reason')!='length':raise ValueError('Actual128token length completion required')
 prompt=value.get('prompt_token_ids') if engine=='vllm' else choice.get('prompt_token_ids');completion=choice.get('token_ids') if engine=='vllm' else choice.get('response_token_ids')
 for ids,n in ((prompt,3968),(completion,128)):
  if not isinstance(ids,list) or len(ids)!=n or any(type(t) is not int or not 0<=t<248320 for t in ids):raise ValueError('Actual reference IDs missing or invalid')
 if prompt!=expected_prompt:raise ValueError('Actual reference prompt IDs differ from frozen3968IDs')
 usage=value.get('usage')
 if not isinstance(usage,dict) or any(type(usage.get(k)) is not int or usage[k]!=v for k,v in (('prompt_tokens',3968),('completion_tokens',128),('total_tokens',4096))):raise ValueError('Reference usage differs')
 if (usage.get('prompt_tokens_details') or {}).get('cached_tokens',0):raise ValueError('Cached prompt invalidates uncached controls')
 message=choice.get('message')
 if not isinstance(message,dict) or message.get('role')!='assistant' or not isinstance(message.get('content'),str) or message.get('reasoning_content'):raise ValueError('Plain no-thinking assistant text required')
 return {'prompt_token_ids':prompt,'completion_token_ids':completion,'text':message['content'],'streamed_ids_observed':False}

def paired_control(current,ordinary):
 for field in ('prompt_token_ids','completion_token_ids','text'):
  if current[field]!=ordinary[field]:raise ValueError('MTP differs from ordinary4K control: '+field)

def main():
 a=argparse.ArgumentParser(description=__doc__);a.add_argument('--engine',choices=('vllm','sglang'),required=True);a.add_argument('--url',required=True);a.add_argument('--corpus-archive',type=Path,required=True);a.add_argument('--output',type=Path,required=True);a.add_argument('--metadata',type=Path,required=True);a.add_argument('--ordinary-controls',type=Path);args=a.parse_args()
 assert args.url=='http://127.0.0.1:'+('18080' if args.engine=='vllm' else '18081') and not args.output.exists() and sha(args.corpus_archive.read_bytes())==ARCHIVE_SHA256
 metadata_raw=args.metadata.read_bytes();metadata=json.loads(metadata_raw)
 assert metadata['engine']==args.engine and metadata['checkpoint_revision']=='482ca0f3832238542f8f5295dde86b5f22711d80' and metadata['context_capacity']==4096
 assert metadata['generation_mode'] in ('ordinary','mtp') and (args.ordinary_controls is not None)==(metadata['generation_mode']=='mtp')
 if args.ordinary_controls is not None:
  previous=json.loads((args.ordinary_controls/'report.json').read_bytes());assert previous['status']=='all124_nonstream_id_controls_collected' and previous['engine']==args.engine and len(previous['cases'])==124
 with zipfile.ZipFile(args.corpus_archive) as z:
  assert z.testzip() is None and len(z.namelist())==len(set(z.namelist()));raw=z.read('manifest.json');assert sha(raw)==MANIFEST_SHA256;corpus=json.loads(raw);assert len(corpus['rows'])==124 and len(corpus['cohorts'])==8;prepared={}
  for row in corpus['rows']:
   wire=z.read(row['request_file']);ids_raw=z.read(row['expected_prompt_ids_file']);assert sha(wire)==row['request_sha256'] and sha(ids_raw)==row['expected_prompt_ids_sha256'];body=json.loads(wire);ids=json.loads(ids_raw)
   assert body['stream'] and body['model']=='qwen38-27b-nvfp4' and body['max_tokens']==128 and len(ids)==3968 and row['total_context_capacity']==4096
   prepared[row['key']]=(body,ids)
 assert len(prepared)==124;out=args.output;out.mkdir();controls=out/'controls';controls.mkdir();actual={};control_report={'status':'running','engine':args.engine,'cases':[],'streamed_ids_observed':False,'scope':'Untimed actual nonstreaming IDs; streaming will check actual counts and exact same-engine greedy text only. No token IDs reconstructed from text.'}
 def save_controls():(controls/'report.json').write_text(json.dumps(control_report,indent=2),encoding='utf8')
 save_controls()
 try:
  for key,(body,ids) in prepared.items():
   wire=json.dumps(control_request(body,args.engine),separators=(',',':')).encode();(controls/(key+'-request.json')).write_bytes(wire)
   try:
    with urlopen(Request(args.url+'/v1/chat/completions',data=wire,headers={'Content-Type':'application/json'}),timeout=300) as response:raw=response.read()
   except HTTPError as error:(controls/(key+'-error.bin')).write_bytes(error.read());raise
   (controls/(key+'-response.json')).write_bytes(raw);actual[key]=control_response(json.loads(raw),args.engine,ids)
   if args.ordinary_controls is not None:
    assert (args.ordinary_controls/(key+'-request.json')).read_bytes()==wire
    paired_control(actual[key],control_response(json.loads((args.ordinary_controls/(key+'-response.json')).read_bytes()),args.engine,ids))
   control_report['cases'].append({'key':key,'request_sha256':sha(wire),'response_sha256':sha(raw),'prompt_ids_exact':True,'output_tokens':128,'ordinary_ids_exact':args.ordinary_controls is not None});save_controls()
  control_report['status']='all124_nonstream_id_controls_collected';save_controls()
 except BaseException as error:control_report.update(status='failed',error=repr(error));save_controls();raise
 run=uuid.uuid4().hex;prefix=run+'-';cohorts=[];sampler=None;manifest={'run_id':run,'status':'running','engine':args.engine,'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'argv':sys.argv,'corpus_archive_sha256':ARCHIVE_SHA256,'corpus_manifest_sha256':MANIFEST_SHA256,'context_capacity':4096,'actual_input_tokens_required':3968,'actual_completion_tokens_required':128,'engine_metadata':metadata,'engine_metadata_sha256':sha(metadata_raw),'cohorts':cohorts,'nonstream_controls_sha256':sha((controls/'report.json').read_bytes()),'streamed_ids_observed':False,'performance_admitted':False,'scope':'Actual frozen4K loopback requests with continuous usage; actual separate prompt/completion-ID controls and same-engine exact text checked. Streamed IDs are unavailable in the common reference protocol. No matched quality/tuning/repetition/2x promotion.'}
 def save():(out/(prefix+'manifest.json')).write_text(json.dumps(manifest,indent=2),encoding='utf8')
 def send(key):
  body,_=prepared[key];result=request(args.url,body,out,prefix+key,128)
  if result['status']!='completed' or result['prompt_tokens']!=3968 or result['generated_tokens']!=128 or result['text']!=actual[key]['text']:raise ValueError('Preserved failed or nonmatching same-engine4Kstream: '+key)
  return result
 save()
 try:
  # Controls reuse the frozen images. Retire their processor/encoder entries
  # before timing; all images within the timed corpus are distinct.
  if args.engine=='vllm':
   reset=reset_idle_caches(args.url,out/'cache-control',model='qwen38-27b-nvfp4');manifest['image_cache_reset']=reset;save()
  sampler=Sampler(out/(prefix+'vram.jsonl'));manifest['gpu']=sampler.device;sampler.start()
  for cohort in corpus['cohorts']:
   case=cohort['case'];c=cohort['concurrency'];sampler.phase='warmup-'+case+'-c'+str(c)
   for wave in cohort['warmups']:
    with concurrent.futures.ThreadPoolExecutor(max_workers=c) as pool:
     for f in [pool.submit(send,key) for key in wave]:f.result()
   phase=case+'-c'+str(c);sampler.phase=phase;sampler.sample()
   with concurrent.futures.ThreadPoolExecutor(max_workers=c) as pool:measured=[f.result() for f in [pool.submit(send,key) for key in cohort['measured']]]
   sampler.sample();summary=summarize(measured);summary.update(sampler.peak(phase));summary.update(case=case,concurrency=c,request_keys=[r['key'] for r in measured],single_token_arrivals=all(r['single_token_arrivals'] for r in measured),prompt_tokens=[r['prompt_tokens'] for r in measured]);cohorts.append(summary);save();print(json.dumps(summary),flush=True)
  if args.engine=='vllm':manifest['image_cache_counter_delta']=verify_after_run(args.url,out/'cache-control',reset['after_reset'])
  manifest['status']='completed'
 except BaseException as error:manifest.update(status='failed',error=repr(error));raise
 finally:
  try:
   if sampler is not None:sampler.close()
  except BaseException as error:manifest.update(status='failed',error=repr(error));raise
  finally:manifest['sampling_errors']=sampler.errors if sampler is not None else [];save()
 return 0
if __name__=='__main__':raise SystemExit(main())
