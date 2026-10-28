"""CPU process/HTTP protocol fixtures only, never model speed measurements."""
import concurrent.futures,json,queue,sys,tempfile,threading,time,unittest
from unittest.mock import patch
from pathlib import Path
from native_batch_http_bridge import BatchWorker,BatchHandler,ThreadingHTTPServer
from benchmark_stream import payload,request

FIXTURE=r"""import json,sys,time
print(json.dumps({'event':'ready','protocol':'study_batch_v1','batch':2,'fixture_only':True}),flush=True)
for line in sys.stdin:
 jobs=json.loads(line)['batch'];print('FIXTURE_BATCH_SIZE '+str(len(jobs)),flush=True)
 for n in (1,2):
  for job in reversed(jobs):
   first=job['body']['messages'][0]['content']=='row0';text=('A' if first else 'B')+('\ufffd' if n==1 else 'é')
   print(json.dumps({'event':'token','id':job['id'],'token_id':(10 if first else 20)+n,'sampled_tokens':n,'prompt_tokens':3,'text':text,'stop':False}),flush=True)
  time.sleep(.01)
 for job in jobs:
  first=job['body']['messages'][0]['content']=='row0';base=10 if first else 20
  print(json.dumps({'event':'completed','id':job['id'],'generated_ids':[base+1,base+2],'prompt_ids':[3,4,5],'text':('A' if first else 'B')+'é','finish_reason':'length'}),flush=True)
"""

class NativeBatchBridgeTests(unittest.TestCase):
 def collector(self,steps,capacity=4,wait=.010):
  worker=BatchWorker.__new__(BatchWorker);worker.capacity=capacity;worker.wait_seconds=wait;worker.stop=threading.Event()
  clock=[100.0];timeouts=[]
  class Jobs:
   def get(self,timeout):
    timeouts.append(timeout);elapsed,value=steps.pop(0);clock[0]+=elapsed
    if value=='empty':raise queue.Empty
    return value
  worker.jobs=Jobs()
  return worker,clock,timeouts
 def test_early_queue_timeout_does_not_close_group_before_qpc_deadline(self):
  worker,clock,timeouts=self.collector([(.002,'empty'),(.001,'second'),(.007,'empty')])
  with patch('native_batch_http_bridge.time.perf_counter',side_effect=lambda:clock[0]):
   group,started,deadline=worker._collect_group('first')
  self.assertEqual(group,['first','second']);self.assertAlmostEqual(deadline-started,.010)
  self.assertEqual(len(timeouts),3);self.assertAlmostEqual(timeouts[1],.008)
 def test_full_group_dispatches_immediately_without_consuming_wait_budget(self):
  worker,clock,timeouts=self.collector([(.001,'second')],capacity=2)
  with patch('native_batch_http_bridge.time.perf_counter',side_effect=lambda:clock[0]):
   group,started,deadline=worker._collect_group('first')
  self.assertEqual(group,['first','second']);self.assertLess(clock[0],deadline);self.assertEqual(len(timeouts),1)
 def test_shutdown_sentinel_stops_partial_group(self):
  worker,clock,timeouts=self.collector([(.001,None)])
  with patch('native_batch_http_bridge.time.perf_counter',side_effect=lambda:clock[0]):
   group,_,_=worker._collect_group('first')
  self.assertEqual(group,['first']);self.assertTrue(worker.stop.is_set())
 def test_zero_wait_never_reads_another_ticket(self):
  worker,clock,timeouts=self.collector([],wait=0)
  with patch('native_batch_http_bridge.time.perf_counter',side_effect=lambda:clock[0]):
   group,_,_=worker._collect_group('first')
  self.assertEqual(group,['first']);self.assertEqual(timeouts,[])
 def worker(self,root,source=FIXTURE):
  p=root/'fixture.py';p.write_text(source,encoding='utf8');worker=BatchWorker([sys.executable,'-u',str(p)],root,2,20)
  deadline=time.monotonic()+5
  while not worker.healthy() and worker.failure is None and time.monotonic()<deadline:time.sleep(.005)
  return worker
 def test_two_http_clients_group_and_receive_only_their_own_tokens(self):
  with tempfile.TemporaryDirectory() as tmp:
   root=Path(tmp);worker=self.worker(root);self.assertTrue(worker.healthy())
   server=ThreadingHTTPServer(('127.0.0.1',0),BatchHandler);server.worker=worker;server.model='qwen38-27b-nvfp4'
   thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
   try:
    barrier=threading.Barrier(2);url='http://127.0.0.1:'+str(server.server_port)
    def call(row):
     body,_=payload('text','fixture',1,'qwen38-27b-nvfp4',2);body['messages'][0]['content']='row'+str(row);barrier.wait(timeout=5)
     return request(url,body,root,'fixture-'+str(row),2)
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:results=list(pool.map(call,[0,1]))
    self.assertEqual([r['text'] for r in results],['Aé','Bé']);self.assertTrue(all(r['status']=='completed' and r['generated_tokens']==2 for r in results))
    self.assertIn('FIXTURE_BATCH_SIZE 2',(root/'worker-stdout.jsonl').read_text(encoding='utf8'))
    dispatch=[json.loads(line) for line in (root/'batch-dispatch.jsonl').read_text().splitlines()]
    self.assertEqual(len(dispatch),1);self.assertEqual(dispatch[0]['real_rows'],2)
    self.assertEqual(len(set(dispatch[0]['request_ids'])),2)
   finally:server.shutdown();server.server_close();thread.join();worker.close()
 def test_partial_group_is_dispatched_without_waiting_for_another_client(self):
  with tempfile.TemporaryDirectory() as tmp:
   root=Path(tmp);worker=self.worker(root)
   try:
    body={'messages':[{'content':'row0'}]};events=list(worker.generate(body,'only'))
    self.assertEqual([e['sampled_tokens'] for e in events if e['event']=='token'],[1,2])
    self.assertEqual(events[-1]['generated_ids'],[11,12]);self.assertIn('FIXTURE_BATCH_SIZE 1',(root/'worker-stdout.jsonl').read_text(encoding='utf8'))
   finally:worker.close()
 def test_wrong_protocol_capacity_never_becomes_healthy(self):
  with tempfile.TemporaryDirectory() as tmp:
   worker=self.worker(Path(tmp),FIXTURE.replace("'batch':2","'batch':1"))
   try:self.assertFalse(worker.healthy());self.assertIn('protocol/capacity',worker.failure)
   finally:worker.close()
 def test_unknown_worker_request_id_fails_pending_client(self):
  with tempfile.TemporaryDirectory() as tmp:
   source=FIXTURE.replace("'id':job['id']","'id':'unknown'");worker=self.worker(Path(tmp),source)
   try:
    with self.assertRaisesRegex(RuntimeError,'unknown request ID'):list(worker.generate({'messages':[{'content':'row0'}]},'known'))
    self.assertFalse(worker.healthy())
   finally:worker.close()
if __name__=='__main__':unittest.main()
