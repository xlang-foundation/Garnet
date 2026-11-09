# SPDX-License-Identifier: Apache-2.0
"""Explicit native compiled-model residency for sequential generation requests."""
import json,os,threading
import garnet
from .runtime import load

class _ModelLease:
    def __init__(self,session,key,model,keep_execution_contexts):
        self.session=session;self.key=key;self.model=model;self.released=False
        self.keep_execution_contexts=keep_execution_contexts
    def forward(self,request):
        if self.released:raise RuntimeError('Model lease is released')
        self.session.check_active();return self.model.forward(request)
    def runtime_status(self):return self.model.runtime_status()
    def release_runtime(self,keep_device_cache=False):
        if self.released:return
        if not isinstance(keep_device_cache,bool):raise ValueError('keep_device_cache must be boolean')
        self.session.check_active()
        try:self.model.trim_runtime(keep_contexts=self.keep_execution_contexts,keep_device_cache=keep_device_cache)
        except Exception:
            self.model.release_runtime();self.session.models.pop(self.key,None);raise
        finally:
            self.released=True;self.session.leased.remove(self.key)

class GenerationSession:
    """Keep native models/engines; optionally retain their GPU execution contexts.

    Enter on the calling thread, pass as model_session, then close explicitly.
    A session owns a bounded set of fixed compiled profiles. Reopen it when
    changing profiles beyond that bound or changing model source/checkpoints.
    The engine budget counts native refitted host weights, not GPU memory.
    """
    def __init__(self,maximum_models=2,engine_weight_budget_bytes=16<<30,keep_execution_contexts=False):
        if not isinstance(maximum_models,int) or isinstance(maximum_models,bool) or maximum_models<1:raise ValueError('maximum_models must be a positive integer')
        if not isinstance(engine_weight_budget_bytes,int) or isinstance(engine_weight_budget_bytes,bool) or not 0<engine_weight_budget_bytes<2**63:raise ValueError('engine weight budget must be a positive int64 byte count')
        if not isinstance(keep_execution_contexts,bool):raise ValueError('keep_execution_contexts must be boolean')
        self.keep_execution_contexts=keep_execution_contexts
        self.maximum_models=maximum_models;self.models={};self.leased=set();self.owner=None;self.closed=False
        self.context_slots={};self.input_builders={}
        self.pool=garnet.cuda_engine_pool();self.pool.configure(engine_weight_budget_bytes)
    def enter(self):
        if self.closed or self.owner is not None:raise RuntimeError('Session is closed or already entered')
        self.pool.enter();self.owner=threading.get_ident();return self
    def check_active(self):
        if self.closed or self.owner!=threading.get_ident():raise RuntimeError('Generation session must be active on the calling thread')
    def prepare_inputs(self,key,factory):
        self.check_active()
        if key not in self.input_builders:
            if len(self.input_builders)>=self.maximum_models:
                raise RuntimeError('Resident input profile limit reached; close this session before changing profiles')
            self.input_builders[key]=factory()
        return self.input_builders[key]
    def load(self,source,entry,cache,shapes,keep_execution_contexts=None,context_slot=None,**options):
        self.check_active()
        if keep_execution_contexts is None:keep_execution_contexts=self.keep_execution_contexts
        if not isinstance(keep_execution_contexts,bool):raise ValueError('keep_execution_contexts must be boolean')
        if context_slot is not None and (not keep_execution_contexts or not isinstance(context_slot,str) or not context_slot):
            raise ValueError('A context slot requires a nonempty name and retained execution contexts')
        key=json.dumps([os.path.abspath(source),entry,os.path.abspath(cache),shapes,options],sort_keys=True)
        if key in self.leased:raise RuntimeError('Compiled model already has an active session lease')
        if key not in self.models and len(self.models)>=self.maximum_models:
            raise RuntimeError('Resident model profile limit reached; close this session before changing profiles')
        if context_slot is not None:
            previous=self.context_slots.get(context_slot)
            if previous is not None and previous!=key and previous in self.models:
                if previous in self.leased:raise RuntimeError('Execution context slot already has an active lease')
                self.models[previous].trim_runtime(keep_contexts=False)
        if key not in self.models:self.models[key]=load(source,entry,cache,shapes,**options)
        if context_slot is not None:
            self.context_slots[context_slot]=key
        self.leased.add(key);return _ModelLease(self,key,self.models[key],keep_execution_contexts)
    def load_component(self,source,entry,cache,shapes,**options):
        return self.load(source,entry,cache,shapes,keep_execution_contexts=False,**options)
    def stats(self):
        return {'models':len(self.models),'maximum_models':self.maximum_models,'active_leases':len(self.leased),'closed':self.closed,'keep_execution_contexts':self.keep_execution_contexts,'engine_pool':json.loads(self.pool.stats_json())}
    def clear_device_cache(self):
        self.check_active()
        if self.leased:raise RuntimeError('Release active leases before clearing their weight caches')
        for model in self.models.values():model.trim_runtime(keep_contexts=self.keep_execution_contexts)
    def close(self):
        if self.closed:return
        if self.leased:raise RuntimeError('Release active model leases before closing the session')
        if self.owner is not None:self.check_active()
        try:
            for model in self.models.values():model.release_runtime()
        finally:
            self.models.clear()
            self.context_slots.clear()
            self.input_builders.clear()
            if self.owner is not None:self.pool.leave();self.owner=None
            self.pool.close();self.closed=True
