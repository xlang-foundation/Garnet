"""Loopback native continuous-batch study bridge; one engine per campaign."""
import argparse,sys
from pathlib import Path
from http.server import ThreadingHTTPServer
from jinja2.sandbox import ImmutableSandboxedEnvironment
from local_coder_batch_transport import BatchWorker,RequestRejected
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'cuda'/'qwen38_5090'))
from native_http_bridge import Handler

class MetalBatchWorker(BatchWorker):
    def __init__(self,command,root,checkpoint,first_token_timeout=3600):
        environment=ImmutableSandboxedEnvironment(trim_blocks=True,lstrip_blocks=True)
        def fail(message):raise ValueError(message)
        environment.globals['raise_exception']=fail
        self.template=environment.from_string((checkpoint/'chat_template.jinja').read_text())
        super().__init__(command,root,first_token_timeout=first_token_timeout)
    def generate(self,body,key):
        if not isinstance(body['messages'][0]['content'],str):raise RequestRejected('Native batch text profile requires plain text')
        try:rendered=self.template.render(messages=body['messages'],add_generation_prompt=True,enable_thinking=False,tools=None)
        except Exception as error:raise RequestRejected(str(error)) from error
        yield from super().generate(dict(rendered_prompt=rendered,max_tokens=body['max_tokens']),key)

class BatchHandler(Handler):
    def do_GET(self):
        if self.path!='/health':self.reply(404,{'error':'Unknown study endpoint'});return
        worker=self.server.worker;healthy=worker.healthy();native=worker.ready_info
        self.reply(200 if healthy else 503,dict(research_only=True,batch=native.get('active_requests'),
            context_capacity=native.get('capacity'),queued_concurrency=True,native_worker=native))

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',type=Path,required=True);parser.add_argument('--log-root',type=Path,required=True)
    parser.add_argument('--first-token-timeout',type=float,default=3600)
    parser.add_argument('--port',type=int,default=8129);parser.add_argument('command',nargs=argparse.REMAINDER)
    args=parser.parse_args();command=args.command[1:] if args.command[:1]==['--'] else args.command
    if not command:parser.error('Explicit native worker command required')
    worker=MetalBatchWorker(command,args.log_root,args.checkpoint,args.first_token_timeout)
    server=ThreadingHTTPServer(('127.0.0.1',args.port),BatchHandler);server.worker=worker;server.model='local-coder-27b'
    try:server.serve_forever()
    finally:server.server_close();worker.close()
if __name__=='__main__':main()
