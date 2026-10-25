"""Synthetic CPU protocol/isolation/accounting controls, no reference execution."""
from pathlib import Path
import copy,importlib,sys,unittest
from .test_fixed_length_reference import sample

class CharacterizationControls(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  sys.path.insert(0,str(Path(__file__).resolve().parent));cls.module=importlib.import_module('reference_concurrency_characterization')
 def test_changed_actual_ids_and_text_are_reported_without_equivalence_claim(self):
  before={'completion_token_ids':[1,2,3],'text':'one'};after={'completion_token_ids':[1,9,3],'text':'other'};old=copy.deepcopy((before,after));result=self.module.compare_outputs(after,before)
  self.assertFalse(result['completion_ids_exact']);self.assertFalse(result['text_exact']);self.assertEqual(result['first_different_token'],1);self.assertFalse(result['cause_or_harmlessness_proven']);self.assertEqual((before,after),old)
 def test_full4k_prompt_and128_ids_still_strictly_required(self):
  value,prompt=sample('sglang');case={'kind':'full4k','expected_prompt_ids':prompt};self.module.validate_id_response(value,case)
  for mutation in ('output','prompt','usage'):
   changed=copy.deepcopy(value)
   if mutation=='output':changed['choices'][0]['response_token_ids']=changed['choices'][0]['response_token_ids'][:-2]
   elif mutation=='prompt':changed['choices'][0]['prompt_token_ids'][20]=9999
   else:changed['usage']['total_tokens']=4094
   with self.subTest(mutation=mutation),self.assertRaises(ValueError):self.module.validate_id_response(changed,case)
 def test_color_isolation_cannot_be_replaced_with_valid_token_counts(self):
  prompt=[248056]*64+[10];case={'kind':'color','key':'red','expected_prompt_ids':prompt,'expected_color':'red'};value={'choices':[{'index':0,'finish_reason':'stop','message':{'role':'assistant','content':'Red'},'prompt_token_ids':prompt,'response_token_ids':[100,248046]}],'usage':{'prompt_tokens':65,'completion_tokens':2,'total_tokens':67}}
  self.module.validate_id_response(value,case);changed=copy.deepcopy(value);changed['choices'][0]['message']['content']='Blue'
  with self.assertRaises(AssertionError):self.module.validate_id_response(changed,case)
 def test_real_overlap_not_declared_thread_count_is_counted(self):
  rows=[{'submitted':1,'completed':4},{'submitted':2,'completed':3},{'submitted':4,'completed':5}];self.assertEqual(self.module.overlap(rows),2)
  self.assertEqual(self.module.overlap([{'submitted':1,'completed':2},{'submitted':2,'completed':3}]),1)
  with self.assertRaises(AssertionError):self.module.overlap([{'submitted':4,'completed':3}])
 def test_raw_stream_requires_consistent_total_usage_and_retains_changed_text(self):
  import json
  from .benchmark_stream import TokenEvents
  case={'key':'synthetic','kind':'full4k','body':{'stream':True,'max_tokens':128}};key='synthetic-stream';body=json.dumps(case['body'],separators=(',',':')).encode();state=TokenEvents();events=[]
  for stamp,n,text,finish in ((1,1,'synthetic ',None),(2,128,'variant','length')):
   raw=json.dumps({'choices':[{'index':0,'delta':{'content':text},'finish_reason':finish}],'usage':{'prompt_tokens':3968,'completion_tokens':n,'total_tokens':3968+n}});state.feed(raw,stamp);events.append({'monotonic':stamp,'data':raw})
  state.feed('[DONE]',3);events.append({'monotonic':3,'data':'[DONE]'});saved=state.result(0,4,128);saved.update(key=key,request_sha256=self.module.sha(body));files={key+'-request.json':body,key+'-result.json':json.dumps(saved).encode(),key+'-events.jsonl':b'\n'.join(json.dumps(v).encode() for v in events)}
  result=self.module.replay_stream_record(files.__getitem__,key,case);self.assertEqual(result['text'],'synthetic variant');self.assertFalse(result['individual_completion_ids_observed'])
  changed=copy.deepcopy(events);event=json.loads(changed[1]['data']);event['usage']['total_tokens']=4094;changed[1]['data']=json.dumps(event);files[key+'-events.jsonl']=b'\n'.join(json.dumps(v).encode() for v in changed)
  with self.assertRaises(AssertionError):self.module.replay_stream_record(files.__getitem__,key,case)
 def test_all288_synthetic_http_requests_replay_and_detect_changed_wave_or_missing_row(self):
  import hashlib,json,tempfile,threading,time
  from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
  from unittest.mock import patch
  from .replay_reference_concurrency_characterization import replay
  groups={};lookup={}
  for group in ('text','image','mixed-color'):
   cases=[]
   for i in range(8):
    color=group=='mixed-color';key=group+'-'+str(i);prompt=([248056]*64+[10+i]) if color else [1]*3967+[100+i]
    body={'model':'synthetic-only','stream':not color,'max_tokens':32 if color else 128,'temperature':0,'request_marker':key}
    if not color:body.update(ignore_eos=True,stream_options={'include_usage':True,'continuous_usage_stats':True})
    case={'key':key,'kind':'color' if color else 'full4k','case':group,'body':body,'expected_prompt_ids':prompt}
    if color:case['expected_color']=('red','green','blue','blue')[i%4]
    cases.append(case);lookup[key]=case
   groups[group]=cases
  lock=threading.Lock();active=[0]
  class Handler(BaseHTTPRequestHandler):
   protocol_version='HTTP/1.1'
   def log_message(self,*args):pass
   def do_POST(self):
    body=json.loads(self.rfile.read(int(self.headers['Content-Length'])));case=lookup[body['request_marker']]
    with lock:active[0]+=1;variant=int(active[0]>1)
    try:
     time.sleep(.02);color=case['kind']=='color';ids=[100+variant,248046] if color else [100+variant]+[2]*127;text=case['expected_color'] if color else 'synthetic variant '+str(variant);prompt=case['expected_prompt_ids'];finish='stop' if color else 'length'
     if body.get('stream'):
      self.send_response(200);self.send_header('Content-Type','text/event-stream');self.send_header('Connection','close');self.end_headers()
      chunks=((len(ids),text,'stop'),) if color else ((1,text[:1],None),(128,text[1:],'length'))
      for n,part,reason in chunks:
       data={'id':'synthetic-only','choices':[{'index':0,'delta':{'content':part},'finish_reason':reason}],'usage':{'prompt_tokens':len(prompt),'completion_tokens':n,'total_tokens':len(prompt)+n}};self.wfile.write(('data: '+json.dumps(data)+'\n\n').encode());self.wfile.flush();time.sleep(.01)
      self.wfile.write(b'data: [DONE]\n\n');self.wfile.flush();self.close_connection=True
     else:
      data={'choices':[{'index':0,'finish_reason':finish,'message':{'role':'assistant','content':text},'prompt_token_ids':prompt,'response_token_ids':ids}],'usage':{'prompt_tokens':len(prompt),'completion_tokens':len(ids),'total_tokens':len(prompt)+len(ids)}};raw=json.dumps(data).encode();self.send_response(200);self.send_header('Content-Type','application/json');self.send_header('Content-Length',str(len(raw)));self.end_headers();self.wfile.write(raw);self.wfile.flush()
    finally:
     with lock:active[0]-=1
  server=ThreadingHTTPServer(('127.0.0.1',0),Handler);server.daemon_threads=True;thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
  try:
   with tempfile.TemporaryDirectory() as tmp:
    output=Path(tmp)/'synthetic-only'
    with patch.object(self.module,'URL','http://127.0.0.1:'+str(server.server_port)),patch.object(self.module,'load_cases',return_value=groups):self.module.collect(None,output,{'synthetic_fixture_only':True})
    files={p.name:p.read_bytes() for p in output.iterdir()};result=replay(files.__getitem__,groups);self.assertEqual(result['actual_nonstream_id_requests'],192);self.assertEqual(result['actual_full4k_streams'],64);self.assertEqual(result['actual_color_streams'],32);self.assertEqual(result['concurrent_color_nonstreams'],48);self.assertEqual(result['concurrent_color_streams'],24);self.assertGreater(result['changed_nonstream_outputs'],0);self.assertFalse(result['performance_admitted'])
    for mutation in ('overlap','missing-row'):
     changed=dict(files);report=json.loads(changed['report.json'])
     if mutation=='overlap':report['waves'][-1]['actual_overlap']=0
     else:report['records'].pop()
     changed['report.json']=json.dumps(report).encode();manifest=json.loads(changed['file-manifest.json']);manifest['files']['report.json']=hashlib.sha256(changed['report.json']).hexdigest();changed['file-manifest.json']=json.dumps(manifest).encode()
     with self.subTest(mutation=mutation),self.assertRaises(AssertionError):replay(changed.__getitem__,groups)
  finally:server.shutdown();server.server_close();thread.join(timeout=5)
if __name__=='__main__':unittest.main()
