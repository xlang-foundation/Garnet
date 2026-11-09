from pathlib import Path
import json
import os
import subprocess
import time

z=Path(__file__).resolve().parent
b=json.loads((z/'bindings1.json').read_text())
assert json.loads((z/'ctest-on1-status.json').read_text())['returncode']==0
assert json.loads((z/'ctest-off1-status.json').read_text())['returncode']==0
state=dict(status='running',scope=b['scope'],waves=[])
def save(): (z/'model-state-status1.json').write_text(json.dumps(state,indent=2)+'\n')
save()
for mode in ['off','on']:
    cmd=[b['xlang'],'-B',b['worker'],b['libraries'][mode]['path'],b['source'],b['model'],str(z/('profile-'+mode+'1.json'))]
    (z/('model-'+mode+'1-command.json')).write_text(json.dumps(cmd,indent=2)+'\n')
    env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',MTL_CAPTURE_ENABLED='0',GARNET_METAL_PROFILE_OPS='0')
    wave=dict(mode=mode,status='running',completed=[],state_witnesses=[])
    state['waves'].append(wave)
    log=z/('model-'+mode+'1.stdout')
    with log.open('x') as out,(z/('model-'+mode+'1.stderr')).open('x') as err:
        p=subprocess.Popen(cmd,stdout=out,stderr=err,env=env,start_new_session=True)
        wave['pid']=p.pid;save();print('Started native model',mode,p.pid,flush=True)
        seen=set()
        while p.poll() is None:
            for line in log.read_text(errors='replace').splitlines():
                if not line.startswith('{') or line in seen:continue
                try:e=json.loads(line)
                except ValueError:continue
                seen.add(line)
                if e['event']=='ready':print('Native ready',mode,flush=True)
                if e['event']=='diagnostic_full_state':
                    wave['state_witnesses'].append(e['invocation']);save();print('Full32layer state witness',mode,e['invocation'],flush=True)
                if e['event']=='completed':
                    wave['completed'].append(dict(id=e['id'],tokens=len(e['generated_ids'])));save();print('Native completed',mode,e['id'],len(e['generated_ids']),flush=True)
                if e['event']=='error':wave['error']=e;save();print('Native reported error',mode,repr(e),flush=True)
            time.sleep(.25)
        rc=p.wait();wave.update(status='complete',returncode=rc);save()
    # Parse final tail after terminal; no child/wave is restarted on observation delay.
    events=[json.loads(l) for l in log.read_text().splitlines() if l.startswith('{')]
    wave['completed']=[dict(id=e['id'],tokens=len(e['generated_ids'])) for e in events if e['event']=='completed']
    wave['state_witnesses']=[e['invocation'] for e in events if e['event']=='diagnostic_full_state']
    errors=[e for e in events if e['event']=='error']
    if errors:wave['errors']=errors
    wave['last_event']=events[-1]['event'] if events else None
    save();print('Native terminal',mode,rc,flush=True)
    assert rc==0 and not errors and len(wave['completed'])==4 and wave['last_event']=='closed'
state['status']='collected_pending_audit';save()
