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
b=load(root/'bindings1.json');assert sha(root/'combined1.metal')==b['shader_sha256']
for p,h in b['assets_sha256'].items():assert sha(Path(p))==h,p
def run(name,cmd):
    path=root/(name+'-status.json');assert not path.exists()
    (root/(name+'-command.json')).write_text(json.dumps(cmd,indent=2)+'\n')
    with (root/(name+'.stdout')).open('x') as out,(root/(name+'.stderr')).open('x') as err:
        p=subprocess.Popen(cmd,stdout=out,stderr=err,env=dict(os.environ,MTL_CAPTURE_ENABLED='0'))
        state=dict(status='running',pid=p.pid,started_ns=time.monotonic_ns());path.write_text(json.dumps(state)+'\n');print(name,'started',p.pid,flush=True)
        code=p.wait();state.update(status='terminal',returncode=code,finished_ns=time.monotonic_ns());path.write_text(json.dumps(state)+'\n')
    print(name,'terminal',code,flush=True)
    if code:raise SystemExit(code)
run('compile1',['xcrun','clang++','-std=c++20','-O2','-fobjc-arc','-I',str(root.parent/'mistral-half-attention-native1/source7/third_party'),str(root/'host1.mm'),'-framework','Foundation','-framework','Metal','-o',str(root/'host1')])
(root/'host-binding1.json').write_text(json.dumps(dict(source_sha256=sha(root/'host1.mm'),binary_sha256=sha(root/'host1'),shader_sha256=sha(root/'combined1.metal')),indent=2)+'\n')
run('correctness1',[str(root/'host1'),'correctness',str(root/'combined1.metal'),b['captures'],str(root/'correctness1'),'forward'])
