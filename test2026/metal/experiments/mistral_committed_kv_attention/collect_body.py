from pathlib import Path
import hashlib
import json
import os
import subprocess
import time

r=Path(__file__).resolve().parent;z=r.parent;b=json.loads((r/'bindings1.json').read_text())
assert json.loads((z/'model-state-audit1.json').read_text())['passed']
def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(4*1024*1024),b''):h.update(block)
    return h.hexdigest()
assert sha(Path(b['body_worker']))==b['body_worker_sha256']
hashes={}
for mode in ['off','on']:
    p=Path(b['libraries'][mode]['path'])/'libgarnet.dylib'
    hashes[str(p)]=sha(p);assert hashes[str(p)]==b['libraries'][mode]['sha256']
for p in [Path(b['body_worker']),Path(b['fixture']),z/'source-audit1.json',Path(b['model'])/'model.safetensors']:
    hashes[str(p)]=sha(p)
assert hashes[str(Path(b['fixture']))]=='6847ca35e1b1dac2bf04bb8e4f2cf358a7f7a0acb4da3fc5b3a1fff4fdbc35ad'
assert hashes[str(Path(b['model'])/'model.safetensors')]=='d7182d886f93a9efa73d6d556ee33ffe839a4e09e15965eeef46f295ab5b503d'
for name,h in json.loads((z/'source-audit1.json').read_text())['candidate_owned_files_sha256'].items():
    assert sha(Path(b['source'])/name)==h
(r/'preflight-hashes1.json').write_text(json.dumps(hashes,indent=2)+'\n')
state=dict(status='running',completed=False,runs=[],scope=b['body_scope'],source_hashing_finished_before_collection=True)
def save(): (r/'collector-status1.json').write_text(json.dumps(state,indent=2)+'\n')
order=[(0,1,0),(0,1,1),(0,4,0),(0,4,1),(1,4,1),(1,4,0),(1,1,1),(1,1,0)]
save()
for cohort,owners,enabled in order:
    mode='on' if enabled else 'off';label=f'cohort{cohort}-owners{owners}-{mode}'
    cmd=[b['xlang'],'-B',b['body_worker'],b['libraries'][mode]['path'],b['source'],b['fixture'],str(r/(label+'.json')),str(owners),str(enabled)]
    run=dict(cohort=cohort,owners=owners,enabled=enabled,label=label,argv=cmd,status='running',started_ns=time.monotonic_ns())
    state['runs'].append(run)
    with (r/(label+'.stdout')).open('x') as out,(r/(label+'.stderr')).open('x') as err:
        p=subprocess.Popen(cmd,stdout=out,stderr=err,env=dict(os.environ,MTL_CAPTURE_ENABLED='0',GARNET_METAL_PROFILE_OPS='0',PYTHONDONTWRITEBYTECODE='1'),start_new_session=True)
        run['pid']=p.pid;save();print('Started serial body',label,p.pid,flush=True)
        rc=p.wait();run.update(status='terminal',returncode=rc,finished_ns=time.monotonic_ns());save()
    print('Finished serial body',label,rc,flush=True)
    if rc!=0:
        state['status']='failed';save();raise RuntimeError(label+' failed; inspect retained logs before continuing')
    data=json.loads((r/(label+'.json')).read_text());assert data['completed']
state.update(status='collected_pending_audit',completed=True);save()
