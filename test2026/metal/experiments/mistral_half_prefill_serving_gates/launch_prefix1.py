"""Preflight and launch one serial uncached/cached correctness campaign."""
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
assert sha(Path(b['library'])/'libgarnet.dylib')==b['library_sha256']
assert sha(root/'tools1/local_coder_batch_worker.py')==b['worker_observer_sha256']
assert sha(Path(b['xlang']))==b['xlang_sha256']
assert sha(root/'fixture2.json')==b['fixture_sha256']
native=root.parent/'mistral-half-attention-native1'
assert sha(native/'provenance7.json')==b['foundation_provenance_sha256']
assert sha(native/'candidate-bindings7.json')==b['native_candidate_binding_sha256']
for name,h in load(native/'candidate-bindings7.json')['files_sha256'].items():assert sha(Path(b['source'])/name)==h,name
for name,h in b['model_files_sha256'].items():assert sha(Path(b['model'])/name)==h,name
paths=[root/'bindings1.json',root/'fixture2.json',root/'profile-template1.json',root/'run_http1.py',root/'audit_http2.py',Path(b['library'])/'libgarnet.dylib',Path(b['xlang'])]
paths+=sorted((root/'tools1').iterdir())+[Path(b['model'])/name for name in b['model_files_sha256']]
hashes={str(p):sha(p) for p in paths if p.is_file()}
(root/'preflight-hashes1.json').write_text(json.dumps(hashes,indent=2)+'\n')
dest=root/'launch1-status.json';assert not dest.exists() and not (root/'status1.json').exists()
cmd=[str(workspace/'out/research/qwen-local-coder/asset-env/bin/python'),'-B',str(root/'run_http1.py')]
(root/'launch1-command.json').write_text(json.dumps(cmd,indent=2)+'\n')
with (root/'collection1.stdout').open('x') as out,(root/'collection1.stderr').open('x') as err:
    child=subprocess.Popen(cmd,stdout=out,stderr=err,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',MTL_CAPTURE_ENABLED='0',GARNET_METAL_PROFILE_OPS='0'))
    status=dict(status='running',pid=child.pid,started_ns=time.monotonic_ns(),scope=b['scope']);dest.write_text(json.dumps(status,indent=2)+'\n');print('Started serial half-prefill cache campaign',child.pid,flush=True)
    code=child.wait();status.update(status='terminal',returncode=code,finished_ns=time.monotonic_ns());dest.write_text(json.dumps(status,indent=2)+'\n')
print('Half-prefill cache campaign terminal',code,flush=True)
raise SystemExit(code)
