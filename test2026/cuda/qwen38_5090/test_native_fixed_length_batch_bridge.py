"""CPU process fixtures test eight-row dispatch, not GPU batching/speed."""
from pathlib import Path
import concurrent.futures,json,sys,tempfile,threading,time,unittest
from unittest.mock import patch
from urllib.request import Request,urlopen
from urllib.error import HTTPError
from native_batch_http_bridge import BatchWorker
from native_fixed_length_batch_http_bridge import FixedLengthBatchWorker,FixedLengthBatchHandler,ThreadingHTTPServer
from native_fixed_length_http_bridge import validate_body

FIXTURE=r'''import json,sys
print(json.dumps({'event':'ready','protocol':'study_batch_v1','batch':8,'fixture_only':True}),flush=True)
for line in sys.stdin:
 for job in reversed(json.loads(line)['batch']):
  value=int(job['body']['row'])
  for n in (1,2):print(json.dumps({'event':'token','id':job['id'],'sampled_tokens':n,'token_id':value*10+n,'prompt_tokens':3,'text':str(value)*n}),flush=True)
  print(json.dumps({'event':'completed','id':job['id'],'generated_ids':[value*10+1,value*10+2],'prompt_ids':[1,2,3],'text':str(value)*2,'ignore_eos':job['body']['ignore_eos'],'finish_reason':'length'}),flush=True)
'''

class Batch8Controls(unittest.TestCase):
    def test_original_bridge_still_refuses_batch8_before_starting_a_process(self):
        with patch('native_http_bridge.subprocess.Popen') as launch:
            with self.assertRaises(ValueError):BatchWorker([],Path('unused'),8)
            launch.assert_not_called()
    def test_explicit_bridge_rejects_unqualified_capacities(self):
        for capacity in (True,1,2,3,16):
            with self.subTest(capacity=capacity),self.assertRaises(ValueError):FixedLengthBatchWorker([],Path('unused'),capacity)
    def test_fixed_length_handler_requires_a_boolean_ignore_eos(self):
        body={'model':'qwen38-27b-nvfp4','temperature':0,'chat_template_kwargs':{'enable_thinking':False},'max_tokens':128,'messages':[{'role':'user','content':'example'}],'ignore_eos':True}
        validate_body(body)
        for value in (1,'true',None):
            with self.subTest(value=value),self.assertRaises(ValueError):validate_body(dict(body,ignore_eos=value))
    def test_eight_clients_route_actual_fixture_events_and_preserve_eos_flag(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);fixture=root/'fixture.py';fixture.write_text(FIXTURE,encoding='utf8')
            worker=FixedLengthBatchWorker([sys.executable,'-u',str(fixture)],root,8,20)
            try:
                deadline=time.monotonic()+5
                while not worker.healthy() and worker.failure is None and time.monotonic()<deadline:time.sleep(.005)
                self.assertTrue(worker.healthy(),worker.failure);barrier=threading.Barrier(8)
                def call(row):
                    barrier.wait(timeout=5);return list(worker.generate({'row':row,'ignore_eos':True},'row'+str(row)))
                with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:results=list(pool.map(call,range(8)))
                for row,events in enumerate(results):
                    self.assertEqual([v['token_id'] for v in events if v['event']=='token'],[row*10+1,row*10+2])
                    self.assertEqual(events[-1]['generated_ids'],[row*10+1,row*10+2]);self.assertIs(events[-1]['ignore_eos'],True)
                groups=[json.loads(line) for line in (root/'batch-dispatch.jsonl').read_text().splitlines()]
                self.assertEqual(sum(g['real_rows'] for g in groups),8)
                self.assertTrue(all(g['physical_capacity']==8 and 1<=g['real_rows']<=8 for g in groups))
                self.assertEqual(sorted(k for g in groups for k in g['request_ids']),['row'+str(i) for i in range(8)])
            finally:worker.close()
    def test_eight_ready_tickets_fill_one_group_without_waiting_for_a_ninth(self):
        worker=FixedLengthBatchWorker.__new__(FixedLengthBatchWorker);worker.capacity=8;worker.wait_seconds=.02;worker.stop=threading.Event()
        class Ready:
            def __init__(self):self.n=0
            def get(self,timeout):self.n+=1;return self.n
        worker.jobs=Ready()
        with patch('native_batch_http_bridge.time.perf_counter',return_value=100):group,_,_=worker._collect_group(0)
        self.assertEqual(group,list(range(8)));self.assertEqual(worker.jobs.n,7)
    def test_real_cpu_http_handler_preserves_boolean_eos_policy_and_ids(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);fixture=root/'fixture.py';fixture.write_text(FIXTURE,encoding='utf8')
            worker=FixedLengthBatchWorker([sys.executable,'-u',str(fixture)],root,8,2)
            server=thread=None
            try:
                deadline=time.monotonic()+5
                while not worker.healthy() and worker.failure is None and time.monotonic()<deadline:time.sleep(.005)
                self.assertTrue(worker.healthy(),worker.failure)
                server=ThreadingHTTPServer(('127.0.0.1',0),FixedLengthBatchHandler);server.worker=worker;server.model='qwen38-27b-nvfp4'
                thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start();url='http://127.0.0.1:'+str(server.server_port)
                with urlopen(url+'/health',timeout=5) as response:health=json.loads(response.read())
                self.assertEqual(health['batch'],8);self.assertEqual(health['gpu_batching_qualification'],'pending');self.assertTrue(health['native_worker']['fixture_only'])
                body={'model':'qwen38-27b-nvfp4','temperature':0,'chat_template_kwargs':{'enable_thinking':False},'max_tokens':2,'messages':[{'role':'user','content':'example'}],'ignore_eos':True,'stream':False,'row':3}
                def send(value):return urlopen(Request(url+'/v1/chat/completions',data=json.dumps(value).encode(),headers={'Content-Type':'application/json'}),timeout=5)
                with send(body) as response:result=json.loads(response.read())
                self.assertEqual(result['choices'][0]['response_token_ids'],[31,32]);self.assertEqual(result['usage']['completion_tokens'],2)
                with self.assertRaises(HTTPError) as caught:send(dict(body,ignore_eos='true'))
                self.assertEqual(caught.exception.code,400);caught.exception.close();self.assertTrue(worker.healthy())
            finally:
                if server is not None:server.shutdown();server.server_close();thread.join()
                worker.close()

if __name__=='__main__':unittest.main()
