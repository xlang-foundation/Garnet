"""Checkpoint-free cancellation tests for the real CUDA orchestration loop."""
import os,sys,threading,time
sys.path.insert(0,os.path.join(os.path.abspath(sys.argv[1]),'runtime'))
import cuda_serving

cancelled=threading.Event()
cuda_serving._lrpc_call_cancelled=cancelled.is_set
lock=threading.RLock()
out=[]
def waiting():
    try:
        with cuda_serving._InferenceRequest(lock):out.append('acquired')
    except TimeoutError:out.append('cancelled')
with lock:
    worker=threading.Thread(target=waiting)
    worker.start();time.sleep(.1);cancelled.set();worker.join(2)
    assert not worker.is_alive() and out==['cancelled'],out
cancelled.clear()

class IO:
    def pack_prompt(self,*args):return [0]
class Decoder:
    def __init__(self):self.io=IO();self.steps=0
    def reset(self):self.steps=0
    def prefill_sequence(self,*args):return [1]
    def step(self,*args):
        self.steps+=1
        if self.steps==3:cancelled.set()
        return [1]
class Model(cuda_serving.Model):
    def __init__(self):
        self.lock=threading.RLock();self.batch=1;self.max_output=32
        self.speculative=None;self.decoder=Decoder();self.pack_prompt=True;self.stops=[-1]
    def _prepare(self,*args):return ([0],[[0,0,0]],[1],1,0,0,[],None)
model=Model()
try:
    model.infer_batch(['fixture'],[''],32)
    raise AssertionError('decode continued after cancellation')
except TimeoutError:pass
assert model.decoder.steps==3,model.decoder.steps
released=[]
def acquire_after_cancel():
    acquired=model.lock.acquire(blocking=False)
    released.append(acquired)
    if acquired:model.lock.release()
worker=threading.Thread(target=acquire_after_cancel)
worker.start();worker.join(2)
assert not worker.is_alive() and released==[True],'cancelled decoder retained model lock'

# Cancellation can arrive just as ownership is granted, before __enter__ returns.
cancelled.clear()
class CancelOnAcquire:
    def __init__(self):self.inner=threading.RLock()
    def acquire(self,**kwargs):
        acquired=self.inner.acquire(**kwargs)
        if acquired:cancelled.set()
        return acquired
    def release(self):self.inner.release()
lock=CancelOnAcquire()
try:
    with cuda_serving._InferenceRequest(lock):raise AssertionError('cancelled request entered')
except TimeoutError:pass
released=[]
def acquire_after_entry_cancel():
    acquired=lock.inner.acquire(blocking=False)
    released.append(acquired)
    if acquired:lock.inner.release()
worker=threading.Thread(target=acquire_after_entry_cancel)
worker.start();worker.join(2)
assert not worker.is_alive() and released==[True],'entry cancellation retained model lock'
print('serving cancellation passed')
