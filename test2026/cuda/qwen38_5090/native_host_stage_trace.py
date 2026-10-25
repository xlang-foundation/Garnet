"""Opt-in request host-wall diagnosis; never isolated GPU or serving metrics."""
from contextlib import contextmanager
from pathlib import Path
import copy,json,time


class NativeHostStageTrace:
    def __init__(self,root,mode,path,*,clock=time.monotonic):
        root=Path(root).resolve();path=Path(path).resolve()
        if mode not in ('ordinary','mtp') or not path.is_relative_to(root/'results'):
            raise ValueError('Explicit study results path and generation mode required')
        self.mode=mode;self.clock=clock;self.current=None;self.active=None;self.ready=None
        info=time.get_clock_info('monotonic')
        self.clock_info={'implementation':info.implementation,'resolution_seconds':info.resolution,
                         'monotonic':info.monotonic,'adjustable':info.adjustable,
                         'explicit_fixture_clock':clock is not time.monotonic}
        self.stream=path.open('x',encoding='utf8');self.closed=False

    def _write(self,value):
        self.stream.write(json.dumps(value,allow_nan=False)+'\n');self.stream.flush()

    def observe(self,event):
        if self.closed:raise ValueError('Closed trace')
        if event['event']=='ready':
            if self.ready is not None or self.current is not None:raise ValueError('Duplicate ready')
            if event['generation_mode']!=self.mode:raise ValueError('Mode mismatch')
            self.ready=copy.deepcopy(event);return
        row=self.current
        if row is None or event.get('id')!=row['id']:raise ValueError('Request event ownership mismatch')
        now=self.clock()
        if event['event']=='token':
            if row['completed'] or event['sampled_tokens']!=row['token_events']+1:
                raise ValueError('Non-contiguous token event')
            row['token_events']+=1
            if row['first_token_observed'] is None:row['first_token_observed']=now
            row['last_token_observed']=now
        elif event['event']=='completed':
            if row['completed'] or not row['token_events'] or len(event['generated_ids'])!=row['token_events']:
                raise ValueError('Completion does not match emitted tokens')
            row['completed']=True;row['completion_observed']=now
            row['prompt_tokens']=len(event['prompt_ids']);row['output_tokens']=len(event['generated_ids'])
            row['finish_reason']=event['finish_reason']
        else:raise ValueError('Unsupported worker event')

    def start(self,key):
        if self.closed or self.ready is None or self.current is not None or not isinstance(key,str) or not key:
            raise ValueError('Open ready trace and new request ID required')
        self.current={'id':key,'mode':self.mode,'started':self.clock(),'phases':[],
                      'first_token_observed':None,'last_token_observed':None,'token_events':0,'completed':False}

    @contextmanager
    def stage(self,name):
        if self.current is None or self.active is not None or not isinstance(name,str) or not name:
            raise ValueError('One flat host stage in an active request required')
        row={'name':name,'started':self.clock(),'status':'running'};self.active=row
        try:yield
        except BaseException as error:
            row['status']='failed';row['error']=repr(error);raise
        else:row['status']='completed'
        finally:
            row['finished']=self.clock();self.current['phases'].append(row);self.active=None

    def finish(self):
        if self.current is None or self.active is not None or not self.current['completed']:
            raise ValueError('Completed request with closed phases required')
        row=self.current;row['finished']=self.clock();row['status']='instrumented_request_completed'
        row['ready']=self.ready;row['clock_info']=self.clock_info
        row['scope']='Instrumented sequential worker host-wall intervals, including Python/native calls and their waits. Phases are flat; no isolated GPU/kernel time, HTTP/queue/transport timing or benchmark/speedup admission.'
        self._write(row);self.current=None

    def close(self):
        if self.closed:return
        if self.active is not None:raise ValueError('Close phase before closing trace')
        if self.current is not None:
            row=self.current;row.update(status='instrumented_request_aborted',finished=self.clock())
            self._write(row);self.current=None
        self.stream.close();self.closed=True
