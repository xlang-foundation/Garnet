"""One fresh native candidate attribution process, with preflight bindings."""
import sys
from pathlib import Path
import hashlib,json,os,subprocess,time
root=Path(sys.argv[1]).resolve() if len(sys.argv)>1 else Path(__file__).resolve().parent
study=root.parent/'mistral-half-attention-native1'
load=lambda p:json.loads(p.read_text())
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()
assert sha(study/'provenance7.json')=='f47d1073955b9943d1038bc066d9fc0d04aff79599a6c951051fcaae74fefe08'
assert load(study/'model-state-audit7.json')['passed'] and load(study/'body7/body-audit7.json')['passed']
b=load(study/'body7/bindings1.json');worker=root/'worker1.py'
paths=[worker,Path(b['libraries']['on']['path'])/'libgarnet.dylib',Path(b['fixture']),Path(b['model'])/'model.safetensors',Path(b['xlang'])]
hashes={str(p):sha(p) for p in paths}
assert hashes[str(paths[1])]==b['libraries']['on']['sha256']
assert hashes[str(paths[3])]=='d7182d886f93a9efa73d6d556ee33ffe839a4e09e15965eeef46f295ab5b503d'
for name,h in load(study/'candidate-bindings7.json')['files_sha256'].items():assert sha(Path(b['source'])/name)==h,name
(root/'preflight1.json').write_text(json.dumps(dict(passed=True,files_sha256=hashes,foundation_provenance_sha256=sha(study/'provenance7.json'),candidate_binding_sha256=sha(study/'candidate-bindings7.json'),diagnostic_only=True),indent=2)+'\n')
cmd=[b['xlang'],'-B',str(worker),b['libraries']['on']['path'],b['source'],b['fixture'],str(root/'diagnostic1.json'),'4','1']
(root/'command1.json').write_text(json.dumps(cmd,indent=2)+'\n')
dest=root/'status1.json';assert not dest.exists()
with (root/'run1.stdout').open('x') as out,(root/'run1.stderr').open('x') as err:
    child=subprocess.Popen(cmd,stdout=out,stderr=err,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',MTL_CAPTURE_ENABLED='0',GARNET_METAL_PROFILE_OPS='0'),start_new_session=True)
    status=dict(status='running',pid=child.pid,diagnostic_only=True,started_ns=time.monotonic_ns());dest.write_text(json.dumps(status)+'\n')
    print('Started instrumented native candidate',child.pid,flush=True)
    code=child.wait();status.update(status='terminal',returncode=code,finished_ns=time.monotonic_ns());dest.write_text(json.dumps(status)+'\n')
print('Instrumented candidate terminal',code,flush=True)
raise SystemExit(code)
