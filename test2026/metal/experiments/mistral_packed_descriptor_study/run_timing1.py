"""Serial opposite-order component timing after independent correctness audit."""
import sys
from pathlib import Path
import hashlib,json,os,subprocess,time
root=Path(sys.argv[1]).resolve() if len(sys.argv)>1 else Path(__file__).resolve().parent
load=lambda p:json.loads(p.read_text())
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()
b=load(root/'bindings1.json');h=load(root/'host-binding1.json')
assert load(root/'correctness-audit1.json')['passed']
hashes={**b['assets_sha256'],str(root/'host1'):h['binary_sha256'],str(root/'host1.mm'):h['source_sha256'],str(root/'combined1.metal'):h['shader_sha256'],str(root/'correctness-audit1.json'):sha(root/'correctness-audit1.json')}
assert all(sha(Path(p))==value for p,value in hashes.items())
(root/'timing-preflight1.json').write_text(json.dumps(dict(passed=True,files_sha256=hashes,hashing_finished_before_collection=True),indent=2)+'\n')
dest=root/'timing-status1.json';assert not dest.exists()
state=dict(status='running',runs=[],hashing_finished_before_collection=True)
def save():dest.write_text(json.dumps(state,indent=2)+'\n')
save()
for order in ['forward','reverse']:
    name='timing-'+order+'1';cmd=[str(root/'host1'),'timing',str(root/'combined1.metal'),b['captures'],str(root/name),order]
    (root/(name+'-command.json')).write_text(json.dumps(cmd,indent=2)+'\n')
    with (root/(name+'.stdout')).open('x') as out,(root/(name+'.stderr')).open('x') as err:
        child=subprocess.Popen(cmd,stdout=out,stderr=err,env=dict(os.environ,MTL_CAPTURE_ENABLED='0'))
        run=dict(order=order,status='running',pid=child.pid,started_ns=time.monotonic_ns(),command=cmd);state['runs'].append(run);save();print(name,'started',child.pid,flush=True)
        code=child.wait();run.update(status='terminal',returncode=code,finished_ns=time.monotonic_ns());save();print(name,'terminal',code,flush=True)
    if code:state['status']='failed';save();raise SystemExit(code)
state['status']='collected_pending_audit';save()
