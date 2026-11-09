"""Loopback native continuous-batch study bridge; one engine per campaign."""
import argparse,json,sys
from pathlib import Path
from http.server import ThreadingHTTPServer
from jinja2.sandbox import ImmutableSandboxedEnvironment
from local_coder_batch_transport import BatchWorker,RequestRejected
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'cuda'/'qwen38_5090'))
from native_http_bridge import Handler

class MetalBatchWorker(BatchWorker):
    def __init__(self,command,root,checkpoint,first_token_timeout=3600,model_family='qwen38_27b'):
        environment=ImmutableSandboxedEnvironment(trim_blocks=True,lstrip_blocks=True)
        def fail(message):raise ValueError(message)
        environment.globals['raise_exception']=fail
        self.special_tokens={}
        if model_family=='mistral_7b':
            metadata=json.loads((checkpoint/'tokenizer_config.json').read_text())
            template=metadata['chat_template']
            if isinstance(template,list):
                if (any(not isinstance(entry,dict) or not isinstance(entry.get('name'),str) or
                        not isinstance(entry.get('template'),str) for entry in template) or
                    len({entry['name'] for entry in template})!=len(template)):
                    raise ValueError('Invalid or duplicate named Mistral chat templates')
                template={entry['name']:entry['template'] for entry in template}
            if isinstance(template,dict):template=template.get('default')
            if not isinstance(template,str):raise ValueError('Explicit default Mistral chat template required')
            for name in ('bos_token','eos_token'):
                if not isinstance(metadata.get(name),str):raise ValueError('Explicit Mistral special-token strings required')
                self.special_tokens[name]=metadata[name]
        elif model_family=='qwen38_27b':template=(checkpoint/'chat_template.jinja').read_text()
        else:raise ValueError('Unsupported native batch model family')
        self.template=environment.from_string(template)
        super().__init__(command,root,first_token_timeout=first_token_timeout)
    def generate(self,body,key):
        if not isinstance(body['messages'][0]['content'],str):raise RequestRejected('Native batch text profile requires plain text')
        try:rendered=self.template.render(messages=body['messages'],add_generation_prompt=True,enable_thinking=False,tools=None,**self.special_tokens)
        except Exception as error:raise RequestRejected(str(error)) from error
        yield from super().generate(dict(rendered_prompt=rendered,max_tokens=body['max_tokens']),key)

class BatchHandler(Handler):
    def validate_request(self,body):
        if getattr(self.server,'model_family','qwen38_27b')!='mistral_7b':
            return Handler.validate_request(self,body)
        validate_mistral_body(body,self.server.model)
    def do_GET(self):
        if self.path!='/health':self.reply(404,{'error':'Unknown study endpoint'});return
        worker=self.server.worker;healthy=worker.healthy();native=worker.ready_info
        self.reply(200 if healthy else 503,dict(research_only=True,batch=native.get('active_requests'),
            context_capacity=native.get('capacity'),queued_concurrency=True,native_worker=native))

def validate_mistral_body(body,model):
    if not isinstance(body,dict) or body.get('model')!=model or body.get('temperature')!=0:
        raise ValueError('Study requires the pinned Mistral model and greedy generation')
    if body.get('chat_template_kwargs',{}) not in ({},{'enable_thinking':False}):
        raise ValueError('Mistral text study has no configurable thinking template')
    if type(body.get('max_tokens')) is not int or not 2<=body['max_tokens']<=1024:
        raise ValueError('Output limit outside Mistral study profile')
    messages=body.get('messages')
    if (not isinstance(messages,list) or len(messages)!=1 or not isinstance(messages[0],dict) or
        messages[0].get('role')!='user' or not isinstance(messages[0].get('content'),str)):
        raise ValueError('Mistral study requires one plain-text user message')
    if any(body.get(name) is not None for name in ('tools','tool_choice','stop','ignore_eos','min_tokens')):
        raise ValueError('Unsupported Mistral study generation override')

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',type=Path,required=True);parser.add_argument('--log-root',type=Path,required=True)
    parser.add_argument('--first-token-timeout',type=float,default=3600)
    parser.add_argument('--model',default='local-coder-27b')
    parser.add_argument('--model-family',choices=('qwen38_27b','mistral_7b'),default='qwen38_27b')
    parser.add_argument('--port',type=int,default=8129);parser.add_argument('command',nargs=argparse.REMAINDER)
    args=parser.parse_args();command=args.command[1:] if args.command[:1]==['--'] else args.command
    if not command:parser.error('Explicit native worker command required')
    worker=MetalBatchWorker(command,args.log_root,args.checkpoint,args.first_token_timeout,args.model_family)
    server=ThreadingHTTPServer(('127.0.0.1',args.port),BatchHandler);server.worker=worker;server.model=args.model;server.model_family=args.model_family
    try:server.serve_forever()
    finally:server.server_close();worker.close()
if __name__=='__main__':main()
