import sys
from pathlib import Path
import hashlib,json,os,subprocess,time
root=Path(sys.argv[1]).resolve() if len(sys.argv)>1 else Path(__file__).resolve().parent
workspace=root.parents[3]
load=lambda p:json.loads(p.read_text())
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()
b=load(root/'bindings1.json')
for p,h in [(Path(b['library'])/'libgarnet.dylib',b['library_sha256']),(root/'tools1/local_coder_batch_worker.py',b['worker_sha256']),(root/'profile1.json',b['profile_sha256']),(root/'fixture1.json',b['fixture_sha256']),
    (root/'tools1/suite.json',b['suite_sha256']),(root/'tools1/evaluate_local_coder_quality.py',b['evaluator_sha256'])]:assert sha(p)==h,str(p)
for name,h in b['model_files_sha256'].items():assert sha(Path(b['model'])/name)==h,name
assert load(root.parent/'mistral-half-prefix-http1/launch1-status.json')['status']=='terminal'
dest=root/'launch1-status.json';assert not dest.exists()
cmd=[str(workspace/'out/research/qwen-local-coder/asset-env/bin/python'),'-B',str(root/'run_quality1.py')]
(root/'launch1-command.json').write_text(json.dumps(cmd,indent=2)+'\n')
with (root/'collection1.stdout').open('x') as out,(root/'collection1.stderr').open('x') as err:
    p=subprocess.Popen(cmd,stdout=out,stderr=err,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',MTL_CAPTURE_ENABLED='0',GARNET_METAL_PROFILE_OPS='0'))
    state=dict(status='running',pid=p.pid,started_ns=time.monotonic_ns());dest.write_text(json.dumps(state,indent=2)+'\n');print('Started half-prefill eight-task coding check',p.pid,flush=True)
    code=p.wait();state.update(status='terminal',returncode=code,finished_ns=time.monotonic_ns());dest.write_text(json.dumps(state,indent=2)+'\n')
print('Half-prefill coding collection terminal',code,flush=True)
raise SystemExit(code)
