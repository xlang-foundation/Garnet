"""CPU failure/lifecycle controls only; no model, server or GPU execution."""
from pathlib import Path
import io,unittest
from .run_vllm_fixed4k import job_path,linux_path,wait_ready

class Process:
    def __init__(self,code=None):self.code=code
    def poll(self):return self.code

class Reply(io.BytesIO):
    status=200

class OwnerControls(unittest.TestCase):
    def test_attempt_name_cannot_escape_fixed_results_folder(self):
        root=Path('D:/CantorAI/5090-study-20261023')
        for name in ('../other','ordinary','vllm-fixed4k-ordinary-attempt0','vllm-fixed4k-ordinary-attempt1/../../other','other-engine-fixed4k-ordinary-attempt1'):
            with self.subTest(name=name),self.assertRaises(ValueError):job_path(root,name)
        self.assertEqual(job_path(root,'vllm-fixed4k-mtp-attempt2'),root/'results/vllm-fixed4k-mtp-attempt2')
    def test_wsl_paths_keep_fixed_drive_and_spaces_literal(self):
        self.assertEqual(linux_path('D:/CantorAI/5090-study-20261023/results/a b/server.log'),'/mnt/d/CantorAI/5090-study-20261023/results/a b/server.log')
        for path in ('C:/unrelated/server.log','relative/server.log'):
            with self.assertRaises(ValueError):linux_path(path)
    def test_dead_owned_server_is_never_replaced_by_an_unrelated_healthy_endpoint(self):
        calls=[]
        with self.assertRaises(RuntimeError):wait_ready(Process(1),'http://127.0.0.1:18080',100,open_url=lambda *a,**k:calls.append(a))
        self.assertEqual(calls,[])
    def test_observation_timeout_does_not_terminate_or_restart_owned_process(self):
        process=Process();pauses=[]
        def unavailable(*args,**kwargs):raise OSError('Not ready')
        with self.assertRaises(TimeoutError):wait_ready(process,'http://127.0.0.1:18080',100,now=lambda:100,pause=pauses.append,open_url=unavailable)
        self.assertIsNone(process.poll());self.assertEqual(pauses,[])
    def test_transient_observation_retries_same_process_then_retains_health_bytes(self):
        calls=[];pauses=[]
        def endpoint(url,timeout):
            calls.append((url,timeout))
            if len(calls)==1:raise OSError('Starting')
            return Reply(b'ready')
        self.assertEqual(wait_ready(Process(),'http://127.0.0.1:18080',100,now=lambda:0,pause=pauses.append,open_url=endpoint),b'ready')
        self.assertEqual(len(calls),2);self.assertEqual(pauses,[5])

if __name__=='__main__':unittest.main()
