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

if __name__=='__main__':unittest.main()
