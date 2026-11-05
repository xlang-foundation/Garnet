"""Loopback single-active Metal study bridge; external concurrency queues.

Uses the existing native-event transport and checkpoint-owned template. Launch
only after all reference workers exit. This is not batch/continuous admission.
"""
import argparse
from pathlib import Path
import sys
from http.server import ThreadingHTTPServer
from jinja2.sandbox import ImmutableSandboxedEnvironment

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'cuda'/'qwen38_5090'))
from native_http_bridge import Worker,Handler


class MetalWorker(Worker):
    def __init__(self,command,root,checkpoint):
        environment=ImmutableSandboxedEnvironment(trim_blocks=True,lstrip_blocks=True)
        def fail(message):raise ValueError(message)
        environment.globals['raise_exception']=fail
        self.template=environment.from_string((checkpoint/'chat_template.jinja').read_text())
        super().__init__(command,root)

    def generate(self,body,key):
        if not isinstance(body['messages'][0]['content'],str):
            raise ValueError('Metal text study requires plain text; vision is unqualified')
        rendered=self.template.render(messages=body['messages'],add_generation_prompt=True,
                                      enable_thinking=False,tools=None)
        yield from super().generate(dict(rendered_prompt=rendered,max_tokens=body['max_tokens']),key)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--log-root',type=Path,required=True)
    parser.add_argument('--port',type=int,default=8129)
    parser.add_argument('command',nargs=argparse.REMAINDER)
    args=parser.parse_args()
    command=args.command[1:] if args.command[:1]==['--'] else args.command
    if not command:parser.error('Explicit native worker command required')
    worker=MetalWorker(command,args.log_root,args.checkpoint)
    server=ThreadingHTTPServer(('127.0.0.1',args.port),Handler)
    server.worker=worker;server.model='local-coder-27b'
    try:server.serve_forever()
    finally:server.server_close();worker.close()


if __name__=='__main__':main()
