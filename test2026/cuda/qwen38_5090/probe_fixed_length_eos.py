"""Actual 4K natural/forced/natural reset and full-ID MTP/stream correctness; no benchmark."""
from pathlib import Path
import argparse,hashlib,json,sys,threading,time,zipfile
from urllib.request import Request,urlopen
from native_fixed_length_http_bridge import Worker,Handler,ThreadingHTTPServer
from benchmark_stream import request as stream_request
from benchmark_fixed_length_native import final_ids,ARCHIVE_SHA256,MANIFEST_SHA256
sha=lambda raw:hashlib.sha256(raw).hexdigest()

def main():
 a=argparse.ArgumentParser(description=__doc__);a.add_argument('--root',type=Path,required=True);a.add_argument('--output',type=Path,required=True);a.add_argument('--mode',choices=('ordinary','mtp'),required=True);a.add_argument('--ordinary',type=Path);args=a.parse_args();root=args.root;out=args.output;m=Path(__file__).resolve().parent
 assert not out.exists() and (args.ordinary is not None)==(args.mode=='mtp');out.mkdir()
 historical=json.loads((m/'evidence/native-natural-eos-failure-reanalysis.json').read_bytes());assert historical['status']=='native_natural_eos_failure_and_complete_screen_replayed' and historical['actual_generated_tokens']==74 and historical['terminal_token_id']==248046
 old_raw=(root/historical['early_eos_events_file']).read_bytes();assert sha(old_raw)==historical['early_eos_events_sha256'];old_values=[json.loads(json.loads(line)['data']) for line in old_raw.splitlines() if json.loads(line)['data']!='[DONE]'];old_final=next(v for v in old_values if 'response_token_ids' in v);old_ids=old_final['response_token_ids'];assert len(old_ids)==74 and old_ids[-1]==248046
 corpus=m/'evidence/fixed-length-4096-corpus.zip';assert sha(corpus.read_bytes())==ARCHIVE_SHA256
 with zipfile.ZipFile(corpus) as z:
  assert sha(z.read('manifest.json'))==MANIFEST_SHA256;manifest=json.loads(z.read('manifest.json'));row=next(r for r in manifest['rows'] if r['key']=='ctx4096-s3-image-c2-r0');body=json.loads(z.read(row['request_file']));prompt=json.loads(z.read(row['expected_prompt_ids_file']));assert prompt==old_final['prompt_token_ids'] and len(prompt)==3968
 cache=json.loads((root/'results/study-ssd-engine-cache/receipt.json').read_bytes())['cache_root'];worker=Worker(['cmd.exe','/d','/c',str(m/'serve_native_fixed_length_worker.bat'),args.mode,cache],out/'worker');server=thread=None;url='http://127.0.0.1:18082';report={'status':'loading','mode':args.mode,'cases':[],'performance_admitted':False,'scope':'Four actual requests per mode. Same image/prompt that naturally stopped at74: normal74, forced128, normal74again and forced128stream. Ordinary/MTP IDs must match exactly; includes real EOS at position73. Correctness only, no benchmark promotion.'}
 def save():(out/'report.json').write_text(json.dumps(report,indent=2),encoding='utf8')
 save()
 try:
  deadline=time.monotonic()+1200
  while not worker.healthy():
   assert worker.process.poll() is None and worker.failure is None,worker.failure
   if time.monotonic()>deadline:raise TimeoutError('Fixed-length worker readiness expired; preserve original process state')
   time.sleep(.25)
  ready=worker.ready_info;assert ready['batch']==1 and ready['capacity']==4096 and ready['generation_mode']==args.mode and ready['explicit_ignore_eos_supported'];report['ready']=ready;save();server=ThreadingHTTPServer(('127.0.0.1',18082),Handler);server.worker=worker;server.model='qwen38-27b-nvfp4';thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start();forced=None
  for label,ignore in (('natural-before',False),('forced',True),('natural-after',False)):
   request=json.loads(json.dumps(body));request['stream']=False;request['ignore_eos']=ignore;wire=json.dumps(request,separators=(',',':')).encode();(out/(label+'-request.json')).write_bytes(wire)
   with urlopen(Request(url+'/v1/chat/completions',data=wire,headers={'Content-Type':'application/json'}),timeout=300) as response:raw=response.read()
   (out/(label+'-response.json')).write_bytes(raw);value=json.loads(raw);choice=value['choices'][0];ids=choice['response_token_ids'];assert choice['prompt_token_ids']==prompt and len(ids)==(128 if ignore else 74) and ids[:74]==old_ids and choice['finish_reason']==('length' if ignore else 'stop') and value['usage']=={'prompt_tokens':3968,'completion_tokens':len(ids),'total_tokens':3968+len(ids)}
   if ignore:forced=ids
   if args.mode=='mtp':assert choice==json.loads((args.ordinary/(label+'-response.json')).read_bytes())['choices'][0]
   report['cases'].append({'label':label,'ignore_eos':ignore,'actual_tokens':len(ids),'prefix74_exact':True,'ordinary_mtp_exact':args.mode=='mtp','response_sha256':sha(raw)});save()
  result=stream_request(url,body,out,'forced-stream',128);assert result['status']=='completed';events=[json.loads(line) for line in (out/'forced-stream-events.jsonl').read_bytes().splitlines()];final=final_ids(events,prompt);assert final['response_token_ids']==forced and forced[:74]==old_ids and forced[73]==248046
  report['cases'].append({'label':'forced-stream','ignore_eos':True,'actual_tokens':128,'prefix74_exact':True,'nonstream_ids_exact':True});report['status']='fixed_length_eos_profile_collected';save()
 except BaseException as error:report.update(status='failed',error=repr(error));save();raise
 finally:
  if server is not None:server.shutdown();server.server_close()
  if thread is not None:thread.join()
  worker.close()
 return 0
if __name__=='__main__':raise SystemExit(main())
