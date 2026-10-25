"""Opt-in B4/B8 fixed-length research bridge; actual GPU qualification pending.

Uses the existing bounded grouping/actual-event routing and the explicit EOS
policy HTTP handler. Default bridge admission remains B2/B4. A matching B8
model worker and whole-model/memory/quality qualification are still required.
"""
import argparse
from pathlib import Path
from native_batch_http_bridge import BatchWorker
from native_fixed_length_http_bridge import Handler,ThreadingHTTPServer

class FixedLengthBatchWorker(BatchWorker):
    supported_capacities=(4,8)

class FixedLengthBatchHandler(Handler):
    def do_GET(self):
        if self.path!='/health':self.reply(404,{'error':'Unknown study endpoint'});return
        worker=self.server.worker;healthy=worker.healthy()
        info={'research_only':True,'batch':worker.capacity,'context_capacity':4096,
              'queued_concurrency':True,'grouping_delay_ms':worker.wait_seconds*1000,
              'grouping_clock':'perf_counter','gpu_batching_qualification':'pending',
              'explicit_ignore_eos_supported':True}
        if healthy:info['native_worker']=worker.ready_info
        self.reply(200 if healthy else 503,info)

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--port',type=int,default=18082)
    p.add_argument('--log-root',type=Path,required=True);p.add_argument('--batch',type=int,choices=(4,8),default=4)
    p.add_argument('--wait-ms',type=float,default=2);p.add_argument('command',nargs=argparse.REMAINDER);a=p.parse_args()
    command=a.command[1:] if a.command[:1]==['--'] else a.command
    if not command:p.error('Matching qualified native batch worker required')
    worker=FixedLengthBatchWorker(command,a.log_root,a.batch,a.wait_ms)
    server=ThreadingHTTPServer(('127.0.0.1',a.port),FixedLengthBatchHandler)
    server.worker=worker;server.model='qwen38-27b-nvfp4'
    try:server.serve_forever()
    finally:server.server_close();worker.close()

if __name__=='__main__':main()
