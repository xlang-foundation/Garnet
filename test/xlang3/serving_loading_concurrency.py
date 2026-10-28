"""A cold model load must not lock out a different ready model's status."""
import os,sys,json,threading,time
import garnet
root=os.path.abspath(sys.argv[1] if len(sys.argv)>1 else os.path.join(os.path.dirname(__file__),'..','..'))
sys.path.insert(0,os.path.join(root,'runtime'))
import cuda_serving
entered=threading.Event();resume=threading.Event();results={}
profile={'batch':1,'deviceId':0,'maxOutputTokens':16}
weights=os.path.join(root,'test','xlang3');xmodel=os.path.join(root,'xModel','qwen35','vl_2b')
class Model:
    def runtime_status(self):return {'ready':True,'state':'ready','batch':1,'profile_json':json.dumps(profile)}
    def release_runtime(self):pass
def create(root,config,name):
    if name=='Qwen3.5-4B':
        entered.set()
        if not resume.wait(10):raise RuntimeError('test did not release model initialization')
    return Model()
cuda_serving.create_serving_model=create
def serve(name):return json.loads(garnet.serve_model(weights,xmodel,'',json.dumps(profile),name))
first=serve('Qwen3.5-2B')
if first.get('error_message')=='requested CUDA device is unavailable':sys.exit(77)
assert first.get('ready'),first
def load():results['load']=serve('Qwen3.5-4B')
worker=threading.Thread(target=load,daemon=True);worker.start()
assert entered.wait(5),'cold load did not start'
started=time.monotonic()
assert json.loads(garnet.serve_status_json('Qwen3.5-2B'))['ready']
assert time.monotonic()-started<2,'cold initialization blocked a ready model'
assert len(json.loads(garnet.list_loaded_models_json())['models'])==1
assert serve('Qwen3.5-4B')['error_code']=='model_loading'
assert json.loads(garnet.stop_serving('Qwen3.5-4B'))['error_code']=='model_loading'
assert json.loads(garnet.stop_serving())['error_code']=='model_loading'
assert json.loads(garnet.serve_status_json('Qwen3.5-2B'))['ready']
resume.set();worker.join(5)
assert not worker.is_alive() and results['load']['ready'],results
assert len(json.loads(garnet.list_loaded_models_json())['models'])==2
assert garnet.stop_serving()
assert json.loads(garnet.list_loaded_models_json())['models']==[]
def fail_once(root,config,name):
    cuda_serving.create_serving_model=create
    raise RuntimeError('fixture constructor failure')
cuda_serving.create_serving_model=fail_once
assert serve('Qwen3.5-4B')['error_code']=='model_load_failed'
assert serve('Qwen3.5-4B')['ready'],'failed initialization retained its reservation'
assert garnet.stop_serving()
print('garnet-serving-loading-concurrency-passed',flush=True)
