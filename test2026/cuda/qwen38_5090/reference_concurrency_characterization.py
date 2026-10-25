"""Untimed SGLang output/isolation study; preserve every observed variation.

Client timestamps attest request overlap and raw stream accounting only.
No latency/throughput summary or harmlessness/numerical-equivalence claim.
"""
from pathlib import Path
import argparse,base64,concurrent.futures,copy,hashlib,io,json,math,sys,threading,time,zipfile
from urllib.request import Request,urlopen
from urllib.error import HTTPError
sys.path.insert(0,str(Path(__file__).resolve().parent))
from benchmark_fixed_length_reference import control_request,control_response
from benchmark_fixed_length_native import ARCHIVE_SHA256,MANIFEST_SHA256
from benchmark_stream import request as stream_request,TokenEvents
from prepare_vlm_reference import reference_tokens
from reference_fixed4k_sglang_reserved_identity import validate as validate_server
sha=lambda raw:hashlib.sha256(raw).hexdigest()
URL='http://127.0.0.1:18081'
PARENT='sglang-fixed4k-ordinary-attempt2'

def validate_id_response(value,case):
 if case['kind']=='full4k':return control_response(value,'sglang',case['expected_prompt_ids'])
 prompt,ids=reference_tokens(value,'sglang');choice=value['choices'][0];usage=value['usage'];text=choice['message']['content']
 assert type(choice.get('index')) is int and choice['index']==0
 assert prompt==case['expected_prompt_ids'] and prompt.count(248056)==64 and len(ids)<=32 and choice['finish_reason']=='stop'
 assert choice['message']['role']=='assistant' and not choice['message'].get('reasoning_content')
 assert all(type(usage.get(k)) is int and usage[k]==n for k,n in (('prompt_tokens',len(prompt)),('completion_tokens',len(ids)),('total_tokens',len(prompt)+len(ids))))
 assert not (usage.get('prompt_tokens_details') or {}).get('cached_tokens',0)
 assert text.strip().lower().rstrip('.')==case['expected_color'],'Concurrent color/spatial isolation failed: '+case['key']
 return {'prompt_token_ids':prompt,'completion_token_ids':ids,'text':text,'streamed_ids_observed':False}

def compare_outputs(current,baseline):
 left=current['completion_token_ids'];right=baseline['completion_token_ids']
 position=next((i for i,(a,b) in enumerate(zip(left,right)) if a!=b),min(len(left),len(right)))
 return {'completion_ids_exact':left==right,'text_exact':current['text']==baseline['text'],'first_different_token':None if left==right else position,'cause_or_harmlessness_proven':False}

def stream_body(case):
 body=copy.deepcopy(case['body'])
 if case['kind']=='color':
  body['stream']=True;body.pop('return_token_ids',None);body.pop('return_prompt_token_ids',None);body['stream_options']={'include_usage':True,'continuous_usage_stats':True}
 return body

def stream_result(state,submitted,completed,case):
 if case['kind']=='full4k':return state.result(submitted,completed,128)
 assert state.done and state.finish_reason=='stop' and state.prompt==len(case['expected_prompt_ids']) and 1<=state.count<=32 and submitted<=state.first<=state.last<=completed
 text=''.join(state.text);assert text.strip().lower().rstrip('.')==case['expected_color'],'Concurrent streamed color/spatial isolation failed: '+case['key']
 return {'status':'completed','submitted':submitted,'first_token':state.first,'last_token':state.last,'completed':completed,'generated_tokens':state.count,'prompt_tokens':state.prompt,'finish_reason':state.finish_reason,'first_arrival_batch_tokens':state.first_batch,'largest_arrival_batch_tokens':state.max_batch,'single_token_arrivals':state.max_batch==1,'text':text,'scope':'Color isolation observation only; all tokens in one arrival are allowed, no decode-rate measurement.'}

def capture_color_stream(case,output,key):
 wire=json.dumps(stream_body(case),separators=(',',':')).encode();(output/(key+'-request.json')).write_bytes(wire);state=TokenEvents();submitted=time.perf_counter();result=None
 with (output/(key+'-events.jsonl')).open('x',encoding='utf8') as events:
  try:
   with urlopen(Request(URL+'/v1/chat/completions',data=wire,headers={'Content-Type':'application/json'}),timeout=300) as response:
    data=[]
    for line in response:
     now=time.perf_counter();text=line.decode('utf8').rstrip('\r\n')
     if text.startswith('data:'):data.append(text[5:].lstrip(' '))
     elif not text and data:
      raw='\n'.join(data);data=[];events.write(json.dumps({'monotonic':now,'data':raw})+'\n');events.flush();state.feed(raw,now)
   result=stream_result(state,submitted,time.perf_counter(),case)
  except BaseException as error:
   if isinstance(error,HTTPError):events.write(json.dumps({'monotonic':time.perf_counter(),'http_status':error.code,'http_error_body':error.read().decode('utf8',errors='replace')})+'\n');events.flush()
   result={'status':'failed','submitted':submitted,'completed':time.perf_counter(),'error':repr(error)}
  result.update(key=key,request_sha256=sha(wire));(output/(key+'-result.json')).write_text(json.dumps(result,indent=2),encoding='utf8')
 return result

def overlap(rows):
 events=[]
 for row in rows:
  a,b=row['submitted'],row['completed'];assert math.isfinite(a) and math.isfinite(b) and a<b
  events.extend(((a,1),(b,-1)))
 live=maximum=0
 for _,delta in sorted(events,key=lambda v:(v[0],v[1])):live+=delta;maximum=max(maximum,live)
 assert live==0
 return maximum

def load_cases(root=None,read=None):
 if read is None:read=lambda name:(root/name).read_bytes()
 raw=read('Garnet/test2026/cuda/qwen38_5090/evidence/fixed-length-4096-corpus.zip');assert sha(raw)==ARCHIVE_SHA256;groups={}
 with zipfile.ZipFile(io.BytesIO(raw)) as z:
  raw=z.read('manifest.json');assert sha(raw)==MANIFEST_SHA256;manifest=json.loads(raw);rows={v['key']:v for v in manifest['rows']}
  for kind in ('text','image'):
   cohort=next(c for c in manifest['cohorts'] if c['case']==kind and c['concurrency']==8);cases=[]
   for key in cohort['measured']:
    row=rows[key];wire=z.read(row['request_file']);ids=z.read(row['expected_prompt_ids_file']);assert sha(wire)==row['request_sha256'] and sha(ids)==row['expected_prompt_ids_sha256']
    body=json.loads(wire);prompt=json.loads(ids);assert len(prompt)==3968 and body['max_tokens']==128 and body['ignore_eos'] is True
    cases.append({'key':key,'kind':'full4k','case':kind,'body':body,'expected_prompt_ids':prompt,'canonical_request_sha256':sha(wire)})
   assert len(cases)==8;groups[kind]=cases
 old=read('results/'+PARENT+'-evidence.zip');proof=json.loads(read('results/'+PARENT+'-verified.json'));assert proof['status']=='reference_fixed4k_failure_preserved' and sha(old)==proof['archive_sha256']
 with zipfile.ZipFile(io.BytesIO(old)) as z:
  prefix='results/'+PARENT+'/client/greedy-controls/correctness/';oracle=json.loads(z.read(prefix+'oracle.json'));colors=[]
  for row in oracle['cases']:
   wire=z.read(prefix+row['name']+'-request.json');response=z.read(prefix+row['name']+'-response.json');assert sha(wire)==row['request_sha256'] and sha(response)==row['response_sha256']
   body=json.loads(wire);encoded=body['messages'][0]['content'][0]['image_url']['url'];assert sha(base64.b64decode(encoded.split(',',1)[1],validate=True))==row['image_sha256']
   case={'key':row['name'],'kind':'color','case':'mixed-color','body':body,'expected_prompt_ids':row['prompt_ids'],'expected_color':row['expected_color'],'canonical_request_sha256':sha(wire)};validate_id_response(json.loads(response),case);colors.append(case)
  assert len(colors)==4
  groups['mixed-color']=[dict(copy.deepcopy(v),key=v['key']+'-copy'+str(i//4)) for i,v in enumerate(colors+colors)]
 return groups

def replay_stream_record(read,key,case):
 wire=read(key+'-request.json');assert wire==json.dumps(stream_body(case),separators=(',',':')).encode()
 saved=json.loads(read(key+'-result.json'));assert saved['key']==key and saved['request_sha256']==sha(wire)
 events=[json.loads(line) for line in read(key+'-events.jsonl').splitlines()];state=TokenEvents();previous=saved['submitted']
 for event in events:
  now=event['monotonic'];assert math.isfinite(now) and previous<=now<=saved['completed'];previous=now
  if event['data']!='[DONE]':
   usage=json.loads(event['data']).get('usage')
   if usage:assert type(usage.get('total_tokens')) is int and usage['total_tokens']==usage['prompt_tokens']+usage['completion_tokens']
  state.feed(event['data'],now)
 actual=stream_result(state,saved['submitted'],saved['completed'],case)
 for field,value in actual.items():assert saved[field]==value
 if case['kind']=='full4k':assert actual['prompt_tokens']==3968 and actual['generated_tokens']==128 and actual['finish_reason']=='length'
 return {'key':key,'case_key':case['key'],'kind':case['kind'],'submitted':saved['submitted'],'completed':saved['completed'],'text':actual['text'],'individual_completion_ids_observed':False}

def id_request(case,output,key,barrier=None):
 wire=json.dumps(control_request(case['body'],'sglang'),separators=(',',':')).encode();(output/(key+'-request.json')).write_bytes(wire)
 if barrier is not None:barrier.wait(timeout=30)
 submitted=time.perf_counter()
 try:
  with urlopen(Request(URL+'/v1/chat/completions',data=wire,headers={'Content-Type':'application/json'}),timeout=300) as response:raw=response.read()
 except HTTPError as error:(output/(key+'-error.bin')).write_bytes(error.read());raise
 completed=time.perf_counter();(output/(key+'-response.json')).write_bytes(raw);parsed=validate_id_response(json.loads(raw),case)
 result={'key':key,'case_key':case['key'],'kind':case['kind'],'request_sha256':sha(wire),'response_sha256':sha(raw),'submitted':submitted,'completed':completed,'prompt_ids_sha256':sha(json.dumps(parsed['prompt_token_ids'],separators=(',',':')).encode()),'completion_token_ids':parsed['completion_token_ids'],'text':parsed['text'],'individual_completion_ids_observed':True}
 (output/(key+'-record.json')).write_text(json.dumps(result,indent=2),encoding='utf8');return result

def collect(root,output,metadata):
 assert not output.exists();groups=load_cases(root);output.mkdir();records=[];waves=[]
 report={'status':'running','engine':'sglang','generation_mode':'ordinary','metadata':metadata,'records':records,'waves':waves,'serial_after_comparisons':{},'benchmark_metrics_collected':False,'physical_gpu_batch_observed':False,'performance_admitted':False,'quality_qualified':False,'scope':'C1/2/4/8 client-overlap, serial-repeat, concurrent actual nonstream-ID and same-frozen-byte stream characterization plus mixed-color/spatial isolation. Output variants are retained, not silently accepted as numerical/quality equivalence or converted to throughput benchmarks.'}
 def save():(output/'report.json').write_text(json.dumps(report,indent=2),encoding='utf8')
 def wave(cases,label,c,stream=False):
  results=[]
  for start in range(0,len(cases),c):
   batch=cases[start:start+c];barrier=threading.Barrier(len(batch)) if len(batch)>1 and not stream else None
   def send(case):
    key=label+'-'+case['key']
    if not stream:return id_request(case,output,key,barrier)
    if case['kind']=='color':capture_color_stream(case,output,key)
    else:stream_request(URL,case['body'],output,key,128)
    return replay_stream_record(lambda n:(output/n).read_bytes(),key,case)
   with concurrent.futures.ThreadPoolExecutor(max_workers=c) as pool:part=[f.result() for f in [pool.submit(send,case) for case in batch]]
   peak=overlap(part);assert peak==len(part),'Requested client-overlap wave not observed'
   waves.append({'label':label,'declared_client_concurrency':c,'actual_overlap':peak,'stream':stream,'keys':[v['key'] for v in part]});records.extend(part);results.extend(part);save()
  return {v['case_key']:v for v in results}
 save()
 try:
  for group,cases in groups.items():
   before=wave(cases,group+'-serial-before',1)
   observed=wave(cases,group+'-c1-stream',1,True)
   for key,v in observed.items():v['serial_control_text_exact']=v['text']==before[key]['text']
   for c in (2,4,8):
    current=wave(cases,group+'-c'+str(c)+'-ids',c)
    for key,v in current.items():v['serial_control_comparison']=compare_outputs(v,before[key])
    rotated=cases[3:]+cases[:3];again=wave(rotated,group+'-c'+str(c)+'-rotated-ids',c)
    for key,v in again.items():v['serial_control_comparison']=compare_outputs(v,before[key]);v['previous_concurrent_comparison']=compare_outputs(v,current[key])
    observed=wave(cases,group+'-c'+str(c)+'-stream',c,True)
    for key,v in observed.items():v['serial_control_text_exact']=v['text']==before[key]['text'];v['previous_concurrent_control_text_exact']=v['text']==current[key]['text']
    save()
   after=wave(cases,group+'-serial-after',1);report['serial_after_comparisons'][group]={key:compare_outputs(v,before[key]) for key,v in after.items()};save()
  assert len(records)==288 and sum(not r['individual_completion_ids_observed'] for r in records)==96
  report['status']='reference_concurrency_output_characterization_collected_needs_replay';save()
 except BaseException as error:report.update(status='failed',error=repr(error));save();raise
 finally:
  files={p.relative_to(output).as_posix():sha(p.read_bytes()) for p in output.rglob('*') if p.is_file()};(output/'file-manifest.json').write_text(json.dumps({'files':files,'scope':report['scope']},indent=2),encoding='utf8')
 return report

def main():
 a=argparse.ArgumentParser(description=__doc__);a.add_argument('--root',type=Path,required=True);a.add_argument('--output',type=Path,required=True);a.add_argument('--metadata',type=Path,required=True);args=a.parse_args()
 metadata=json.loads(args.metadata.read_bytes());assert metadata['engine']=='sglang' and metadata['generation_mode']=='ordinary' and metadata['configured_engine_context_capacity']==4098 and metadata['source_after_startup_verified'] is True and metadata['owned_server_pid_verified'] is True
 with urlopen(URL+'/get_server_info',timeout=60) as response:info=json.loads(response.read())
 expected=validate_server(info,'ordinary')
 for key,value in expected.items():assert metadata[key]==value
 result=collect(args.root,args.output,metadata);print(json.dumps({'status':result['status'],'observed_requests':len(result['records']),'physical_gpu_batch_observed':False,'performance_admitted':False}));return 0
if __name__=='__main__':raise SystemExit(main())
