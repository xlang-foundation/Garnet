"""Untimed serial native model witnesses, retaining terminal and failure state."""
import sys
from pathlib import Path
import hashlib,json,os,subprocess,time
root=Path(sys.argv[1]).resolve() if len(sys.argv)>1 else Path(__file__).resolve().parent
b=json.loads((root/'model-bindings7.json').read_text())
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
    return h.hexdigest()
for p,h in [(b['worker'],b['worker_sha256']),(b['xlang'],b['xlang_sha256']),(b['fixture'],b['fixture_sha256']),
    (root/'source-audit7.json',b['source_audit_sha256']),(root/'trained-audit7.json',b['trained_audit_sha256'])]:assert sha(p)==h,str(p)
for name,h in b['model_files_sha256'].items():assert sha(Path(b['model'])/name)==h,name
for mode in ['off','on']:
    e=b['libraries'][mode];assert sha(Path(e['path'])/'libgarnet.dylib')==e['sha256']
    assert sha(e['profile'])==e['profile_sha256']
    assert json.loads((root/('suite-'+mode+'3-status.json')).read_text())['returncode']==0
dest=root/'model-state-status7.json';assert not dest.exists()
state=dict(status='running',scope=b['scope'],waves=[])
def save():dest.write_text(json.dumps(state,indent=2)+'\n')
save()
try:
    for mode in ['off','on']:
        entry=b['libraries'][mode]
        cmd=[b['xlang'],'-B',b['worker'],entry['path'],b['source'],b['model'],entry['profile']]
        (root/('model-'+mode+'7-command.json')).write_text(json.dumps(cmd,indent=2)+'\n')
        env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',MTL_CAPTURE_ENABLED='0',GARNET_METAL_PROFILE_OPS='0')
        wave=dict(mode=mode,status='running',completed=[],state_witnesses=[])
        state['waves'].append(wave)
        log=root/('model-'+mode+'7.stdout')
        with log.open('x') as out,(root/('model-'+mode+'7.stderr')).open('x') as err:
            child=subprocess.Popen(cmd,stdout=out,stderr=err,env=env,start_new_session=True)
            wave['pid']=child.pid;save();print('Started native model',mode,child.pid,flush=True)
            offset=0
            def consume():
                global offset
                with log.open() as incoming:
                    incoming.seek(offset)
                    while True:
                        position=incoming.tell();line=incoming.readline()
                        if not line or not line.endswith('\n'):
                            offset=position;return
                        offset=incoming.tell()
                        if not line.startswith('{'):continue
                        event=json.loads(line)
                        kind=event.get('event')
                        if kind=='ready':print('Native ready',mode,flush=True)
                        if kind=='diagnostic_full_state':
                            wave['state_witnesses'].append(event['invocation']);save();print('Full32layer state witness',mode,event['invocation'],flush=True)
                        if kind=='completed':
                            wave['completed'].append(dict(id=event['id'],tokens=len(event['generated_ids'])));save();print('Native completed',mode,event['id'],len(event['generated_ids']),flush=True)
                        if kind=='error':wave.setdefault('errors',[]).append(event);save();print('Native reported error',mode,event,flush=True)
            while child.poll() is None:
                consume();time.sleep(.25)
            consume();code=child.wait();wave.update(status='terminal',returncode=code);save()
        events=[json.loads(line) for line in log.read_text().splitlines() if line.startswith('{')]
        wave['last_event']=events[-1]['event'] if events else None;save();print('Native terminal',mode,code,flush=True)
        assert code==0 and not wave.get('errors') and len(wave['completed'])==4 and wave['last_event']=='closed'
    state['status']='collected_pending_audit';save()
except BaseException as error:
    state.update(status='failed',error=str(error));save();raise
