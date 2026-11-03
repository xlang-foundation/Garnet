"""CPU-only SSE/routing test with a fake worker; never performance evidence."""
import concurrent.futures,json,sys,tempfile,threading,time,unittest,urllib.request,urllib.error
from http.server import ThreadingHTTPServer
from pathlib import Path
from local_coder_batch_transport import BatchWorker
from local_coder_batch_http import BatchHandler
from test_local_coder_batch_transport import FAKE
class HTTPTest(unittest.TestCase):
 def test_parallel_sse_cache_accounting_and_nonfatal_rejection(self):
  with tempfile.TemporaryDirectory() as temp:
   worker=BatchWorker([sys.executable,'-u','-c',FAKE],Path(temp),event_timeout=3)
   server=ThreadingHTTPServer(('127.0.0.1',0),BatchHandler);server.worker=worker;server.model='local-coder-27b'
   thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
   try:
    deadline=time.monotonic()+5
    while not worker.healthy() and time.monotonic()<deadline:time.sleep(.01)
    self.assertTrue(worker.healthy())
    url='http://127.0.0.1:'+str(server.server_port)
    body=dict(model=server.model,temperature=0,chat_template_kwargs={'enable_thinking':False},max_tokens=2,messages=[dict(role='user',content='test')],stream=True)
    def call():
     request=urllib.request.Request(url+'/v1/chat/completions',data=json.dumps(body).encode(),headers={'Content-Type':'application/json'})
     with urllib.request.urlopen(request,timeout=5) as response:return response.read().decode()
    with concurrent.futures.ThreadPoolExecutor(2) as executor:
     responses=list(executor.map(lambda _:call(),range(2)))
    identities=[]
    for response in responses:
     events=[json.loads(line[6:]) for line in response.splitlines() if line.startswith('data: ') and line!='data: [DONE]']
     self.assertEqual(len(events),3);self.assertTrue(response.endswith('data: [DONE]\n\n'))
     identities.append(events[0]['id']);self.assertEqual({e['id'] for e in events},{events[0]['id']})
     self.assertEqual(events[-1]['usage']['completion_tokens'],2)
     self.assertEqual(events[-1]['usage']['prompt_tokens_details']['cached_tokens'],1)
     self.assertEqual(events[-1]['response_token_ids'],[event['token_id'] for event in events[:-1]])
    self.assertNotEqual(*identities)
    self.assertIn('profile rejected',call())
    with urllib.request.urlopen(url+'/health',timeout=5) as response:self.assertEqual(response.status,200)
   finally:
    server.shutdown();server.server_close();worker.close();thread.join(timeout=2)
   self.assertEqual(worker.process.returncode,0)
if __name__=='__main__':unittest.main()
