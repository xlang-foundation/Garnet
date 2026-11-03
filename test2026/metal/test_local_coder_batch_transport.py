"""CPU-only transport checks; fake events are never benchmark evidence."""
import concurrent.futures,json,sys,tempfile,time,unittest
from pathlib import Path
from local_coder_batch_transport import BatchWorker,RequestRejected

FAKE='''import json,sys
print(json.dumps(dict(event="ready",active_requests=2)),flush=True)
jobs=[json.loads(sys.stdin.readline()) for _ in range(2)]
for i in range(2):
 for j,job in enumerate(reversed(jobs)):
  print(json.dumps(dict(event="token",id=job["id"],token_id=j*10+i,sampled_tokens=i+1,prompt_tokens=1,cached_tokens=1,text="x"*(i+1))),flush=True)
for j,job in enumerate(reversed(jobs)):
 print(json.dumps(dict(event="completed",id=job["id"],generated_ids=[j*10,j*10+1],prompt_ids=[7],cached_tokens=1,text="xx",finish_reason="length")),flush=True)
for line in sys.stdin:
 job=json.loads(line)
 print(json.dumps(dict(event="error",id=job["id"],fatal=False,error="profile rejected")),flush=True)
'''
class TransportTest(unittest.TestCase):
 def test_interleaved_routing_rejection_and_cleanup(self):
  with tempfile.TemporaryDirectory() as temp:
   worker=BatchWorker([sys.executable,'-u','-c',FAKE],Path(temp),event_timeout=3)
   try:
    deadline=time.monotonic()+5
    while not worker.healthy() and time.monotonic()<deadline:time.sleep(.01)
    self.assertTrue(worker.healthy())
    with concurrent.futures.ThreadPoolExecutor(2) as executor:
     pending=[executor.submit(lambda key:list(worker.generate({},key)),key) for key in ('A','B')]
     results=[item.result(timeout=5) for item in pending]
    for key,events in zip(('A','B'),results):
     self.assertEqual([event['id'] for event in events],[key]*3)
     self.assertEqual(events[-1]['generated_ids'],[event['token_id'] for event in events[:-1]])
    self.assertTrue(worker.healthy());self.assertFalse(worker.pending)
    with self.assertRaises(RequestRejected):list(worker.generate({},'C'))
    worker.request_failed(RequestRejected('profile'))
    self.assertTrue(worker.healthy());self.assertFalse(worker.pending)
   finally:worker.close()
   self.assertEqual(worker.process.returncode,0)

 def test_queued_first_token_wait_is_separate_from_active_idle(self):
  for first_delay,idle_delay,passes in ((.15,0,True),(0,.15,False)):
   fake="""import json,sys,time
print(json.dumps(dict(event='ready')),flush=True)
job=json.loads(sys.stdin.readline());key=job['id']
time.sleep(FIRST)
print(json.dumps(dict(event='token',id=key,token_id=1,sampled_tokens=1,prompt_tokens=1)),flush=True)
time.sleep(IDLE)
print(json.dumps(dict(event='token',id=key,token_id=2,sampled_tokens=2,prompt_tokens=1)),flush=True)
print(json.dumps(dict(event='completed',id=key,generated_ids=[1,2],prompt_ids=[7])),flush=True)
""".replace('FIRST',str(first_delay)).replace('IDLE',str(idle_delay))
   with tempfile.TemporaryDirectory() as temp:
    worker=BatchWorker([sys.executable,'-u','-c',fake],Path(temp),event_timeout=.05,first_token_timeout=1)
    try:
     deadline=time.monotonic()+3
     while not worker.healthy() and time.monotonic()<deadline:time.sleep(.005)
     self.assertTrue(worker.healthy())
     if passes:self.assertEqual(len(list(worker.generate({},'A'))),3)
     else:
      with self.assertRaises(TimeoutError):list(worker.generate({},'A'))
    finally:worker.close()

if __name__=='__main__':unittest.main()
