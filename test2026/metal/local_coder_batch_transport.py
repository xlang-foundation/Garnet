"""Bounded CPU event routing for one native worker with multiple live requests."""
import json,queue,subprocess,threading,time
from pathlib import Path

class RequestRejected(ValueError):pass

class BatchWorker:
    def __init__(self,command,root,max_pending=16,event_timeout=300):
        self.root=Path(root);self.root.mkdir(parents=True,exist_ok=True)
        self.lock=threading.RLock();self.write_lock=threading.Lock();self.pending={}
        self.max_pending=max_pending;self.event_timeout=event_timeout
        self.ready=False;self.ready_info={};self.failure=None;self.closed=False
        self.stderr=(self.root/'worker-stderr.log').open('ab',buffering=0)
        self.process=subprocess.Popen(command,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=self.stderr,bufsize=0)
        self.reader=threading.Thread(target=self._read,daemon=True);self.reader.start()

    def _fail(self,error):
        with self.lock:
            self.failure=str(error);self.ready=False
            for target in self.pending.values():target.put({'event':'worker_exit','error':self.failure})

    def _read(self):
        try:
            with (self.root/'worker-stdout.jsonl').open('a',encoding='utf-8') as raw:
                for line in self.process.stdout:
                    text=line.decode('utf-8',errors='replace').rstrip('\r\n')
                    raw.write(json.dumps({'received':time.monotonic(),'line':text})+'\n');raw.flush()
                    try:event=json.loads(text)
                    except ValueError:continue
                    if not isinstance(event,dict):continue
                    kind=event.get('event')
                    if kind=='ready':
                        with self.lock:
                            if self.failure is None:self.ready_info=event;self.ready=True
                    elif kind in ('token','completed','error'):
                        with self.lock:target=self.pending.get(event.get('id'))
                        if target is None:raise ValueError('Native event has no live request identity')
                        target.put(event)
        except Exception as error:self._fail(repr(error))
        finally:self._fail(self.failure or 'Native worker exited')

    def healthy(self):
        with self.lock:return self.ready and self.failure is None and self.process.poll() is None and not self.closed

    def request_failed(self,error):
        if not isinstance(error,RequestRejected):self._fail(repr(error))

    def _write(self,value):
        wire=(json.dumps(value,separators=(',',':'))+'\n').encode()
        with self.write_lock:
            while wire:
                count=self.process.stdin.write(wire)
                if not count:raise OSError('Native worker input closed')
                wire=wire[count:]
            self.process.stdin.flush()

    def generate(self,body,key):
        with self.lock:
            if not self.healthy():raise RuntimeError('Native batch worker is not ready')
            if key in self.pending:raise RequestRejected('Duplicate live request identity')
            if len(self.pending)>=self.max_pending:raise RequestRejected('Native pending request limit reached')
            target=queue.Queue();self.pending[key]=target
        terminal=False
        try:
            self._write({'id':key,'body':body});count=0;ids=[];prompt=None;cached=None
            while True:
                try:event=target.get(timeout=self.event_timeout)
                except queue.Empty:raise TimeoutError('Native batch request event deadline')
                kind=event['event']
                if kind=='worker_exit':raise RuntimeError(str(event['error']))
                if event.get('id')!=key:raise ValueError('Native event routing mismatch')
                if kind=='error':
                    terminal=True
                    if event.get('fatal',True):raise RuntimeError(event['error'])
                    raise RequestRejected(event['error'])
                if kind=='token':
                    count+=1;ids.append(event['token_id'])
                    if event['sampled_tokens']!=count:raise ValueError('Native batch token count mismatch')
                    if prompt is None:prompt=event['prompt_tokens']
                    if prompt!=event['prompt_tokens']:raise ValueError('Native batch prompt count changed')
                    reuse=event.get('cached_tokens',0)
                    if type(reuse) is not int or not 0<=reuse<=prompt:raise ValueError('Invalid native cached-token count')
                    if cached is None:cached=reuse
                    if cached!=reuse:raise ValueError('Native cached-token count changed')
                elif kind=='completed':
                    if event['generated_ids']!=ids or len(event['prompt_ids'])!=prompt:
                        raise ValueError('Native batch final IDs/counts mismatch')
                    if event.get('cached_tokens',0)!=cached:raise ValueError('Final cached-token count changed')
                    terminal=True
                yield event
                if terminal:return
        except RequestRejected:raise
        except Exception as error:self._fail(repr(error));raise
        finally:
            with self.lock:self.pending.pop(key,None)
            # An abandoned stream cannot silently leave native work running.
            # The HTTP handler drains disconnected streams, so this is a fatal
            # exceptional caller path rather than normal client cancellation.
            if not terminal:self._fail('Native batch stream abandoned before completion')

    def close(self):
        with self.lock:
            if self.closed:return
            self.closed=True;self.ready=False
        with self.write_lock:self.process.stdin.close()
        try:
            self.process.wait(timeout=120)
            (self.root/'worker-exit.json').write_text(json.dumps(dict(returncode=self.process.returncode,expected_close=True))+'\n')
        finally:
            self.reader.join(timeout=10)
            if self.process.poll() is not None:self.process.stdout.close();self.stderr.close()
