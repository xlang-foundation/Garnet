"""Buffered nested host API intervals, never GPU kernel or serving timings."""
import functools
import threading
import time


class WarmHostTrace:
    def __init__(self, clock=time.perf_counter):
        self.clock=clock;self.active=False;self.stack=[];self.records=[];self.originals=[]
        self.tag=None;self.owner_thread=None;self.sequence=0

    def replace(self,owner,name,method):
        owned=name in owner.__dict__;original=owner.__dict__.get(name)
        self.originals.append((owner,name,original,owned));setattr(owner,name,method)

    def instrument(self, owner, name, stage):
        method=getattr(owner,name)
        @functools.wraps(method)
        def wrapped(instance,*args,**kwargs):
            if not self.active:return method(instance,*args,**kwargs)
            if threading.get_ident()!=self.owner_thread:
                raise RuntimeError('Diagnostic trace crossed Python threads')
            number=self.sequence;self.sequence+=1
            parent=self.stack[-1] if self.stack else None
            started=self.clock();self.stack.append(number);error=None
            try:return method(instance,*args,**kwargs)
            except BaseException as failure:error=repr(failure);raise
            finally:
                ended=self.clock();assert self.stack.pop()==number
                self.records.append({'tag':self.tag,'number':number,'parent':parent,
                    'stage':stage,'started':started,'ended':ended,
                    'host_wall_seconds':ended-started,'error':error})
        self.replace(owner,name,wrapped)

    def begin(self,tag):
        if self.active or self.stack:raise RuntimeError('Nested trace capture is invalid')
        self.tag=tag;self.owner_thread=threading.get_ident();self.active=True

    def end(self):
        if not self.active or self.stack:raise RuntimeError('Unbalanced trace capture')
        self.active=False;self.owner_thread=None;self.tag=None

    def restore(self):
        if self.active or self.stack:raise RuntimeError('Cannot restore active trace')
        for owner,name,method,owned in reversed(self.originals):
            if owned:setattr(owner,name,method)
            else:delattr(owner,name)
        self.originals=[]

    def summaries(self):
        by_number={r['number']:r for r in self.records}
        assert len(by_number)==len(self.records)
        child_time={number:0.0 for number in by_number}
        for record in self.records:
            assert record['ended']>=record['started']
            if record['parent'] is not None:
                parent=by_number[record['parent']]
                assert parent['tag']==record['tag'] and parent['started']<=record['started']<=record['ended']<=parent['ended']
                child_time[parent['number']]+=record['host_wall_seconds']
        result={}
        for record in self.records:
            key=(record['tag'],record['stage'])
            row=result.setdefault(key,{'tag':key[0],'stage':key[1],'calls':0,'inclusive_host_wall_seconds':0.0,'exclusive_host_wall_seconds':0.0,'errors':0})
            exclusive=record['host_wall_seconds']-child_time[record['number']]
            assert exclusive>=-1e-9
            row['calls']+=1;row['inclusive_host_wall_seconds']+=record['host_wall_seconds'];row['exclusive_host_wall_seconds']+=max(0.0,exclusive)
            row['errors']+=record['error'] is not None
        return list(result.values())
