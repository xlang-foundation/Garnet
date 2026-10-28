"""Optional study instrumentation; host intervals are not CUDA kernel timings.

No model imports or automatic installation. Keep absolute clock domains separate.
Profiled requests are diagnostic observations, never the final throughput result.
"""
import re,time


class StageRecorder:
    def __init__(self,emit,clock=time.monotonic):
        self.emit=emit;self.clock=clock;self.active=None;self.sequence=0;self.next_span=0;self.stack=[]

    def _event(self,phase,label,span,parent,ok=None):
        self.sequence+=1
        value=dict(event='study_stage',id=self.active,sequence=self.sequence,phase=phase,stage=label,
                   span=span,parent_span=parent,native_monotonic_seconds=self.clock(),clock_domain='native_worker',
                   measurement='host_wall_interval_including_instrumentation',CUDA_kernel_elapsed_claimed=False)
        if ok is not None:value['call_succeeded']=ok
        self.emit(value)

    def start_request(self,key):
        if self.active is not None or self.stack:raise ValueError('Previous request is still active')
        if not isinstance(key,str) or not re.fullmatch('[0-9a-f]{32}',key):raise ValueError('Expected study HTTP request ID')
        self.active=key;self.sequence=0
        return self.begin('worker_request')

    def begin(self,label):
        if self.active is None or not isinstance(label,str) or not re.fullmatch('[a-z][a-z0-9_]{0,63}',label):raise ValueError('Active request and fixed stage label required')
        self.next_span+=1;span=self.next_span;parent=self.stack[-1][0] if self.stack else None
        self.stack.append((span,label,parent));self._event('begin',label,span,parent);return span

    def end(self,span,ok=True):
        if type(span) is not int or not self.stack or self.stack[-1][0]!=span or type(ok) is not bool:raise ValueError('Stages must close in reverse order')
        _,label,parent=self.stack.pop();self._event('end',label,span,parent,ok)

    def finish_request(self,ok=True):
        if len(self.stack)!=1 or self.stack[0][1]!='worker_request':raise ValueError('Nested stages remain open')
        self.end(self.stack[0][0],ok);self.active=None

    def wrap(self,call,label):return TimedCall(call,label,self)


class TimedCall:
    def __init__(self,call,label,recorder):self.call=call;self.label=label;self.recorder=recorder
    def __call__(self,*args,**kwargs):
        if self.recorder.active is None:return self.call(*args,**kwargs)
        span=self.recorder.begin(self.label);ok=False
        try:
            result=self.call(*args,**kwargs);ok=True;return result
        finally:self.recorder.end(span,ok)


class TimedBridgeLock:
    """Wrap the bridge's existing lock, preserving the serialized request lifetime.

    emit must serialize writes safely across HTTP threads. key reads the calling
    thread's request ID. These timestamps share only the bridge clock domain.
    """
    def __init__(self,lock,key,emit,clock=time.monotonic):self.lock=lock;self.key=key;self.emit=emit;self.clock=clock
    def __enter__(self):
        key=self.key();queued=self.clock();self.emit(dict(event='study_bridge_stage',id=key,phase='queued',bridge_monotonic_seconds=queued,clock_domain='http_bridge'))
        self.lock.acquire()
        try:self.emit(dict(event='study_bridge_stage',id=key,phase='acquired',bridge_monotonic_seconds=self.clock(),clock_domain='http_bridge'))
        except BaseException:self.lock.release();raise
        return self
    def __exit__(self,kind,value,traceback):
        try:self.emit(dict(event='study_bridge_stage',id=self.key(),phase='released',bridge_monotonic_seconds=self.clock(),clock_domain='http_bridge',call_succeeded=kind is None))
        finally:self.lock.release()
        return False
