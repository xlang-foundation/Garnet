from pathlib import Path
import hashlib,json,os,subprocess,time
r=Path(__file__).resolve().parent
assert json.loads((r/'compile-status3.json').read_text())['returncode']==0
assert json.loads((r/'correctness4/audit1.json').read_text())['passed']
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
refs=json.loads((r/'source-references1.json').read_text())
for p,h in refs.items():assert sha(Path(p))==h,p
files=[r/'strict_attention3',r/'strict_attention3.mm',r/'combined6.metal',r/'strict-register2.metal',r/'coordinate-probe1.metal',r/'audit_register_attention1.py',r/'correctness4/correctness1.json',r/'correctness4/audit1.json']+list((r/'correctness4').glob('*.f32'))
preflight={str(p.relative_to(r)):sha(p) for p in files};(r/'register-screen-preflight-hashes1.json').write_text(json.dumps(preflight,indent=2)+'\n')
state=dict(status='running',runs=[],source_hashing_finished_before_collection=True,scope='Two fresh strictly serial processes, opposite rotating triple orders, all variants/shapes3warmups before first timing;9measured triples/cell. Strict FP32 MPP register-softmax attention against original4product and same-cache no-copy2product MPS. Synthetic operator only; no model/serving/quality claim.')
def save():(r/'register-screen-status1.json').write_text(json.dumps(state,indent=2)+'\n')
save()
for cohort,mode in [(0,'screen'),(1,'screen_reverse')]:
 output=r/('register-screen'+str(cohort+1));cmd=[str(r/'strict_attention3'),str(r/'combined6.metal'),str(r/'correctness4'),str(output),mode]
 label='register-screen'+str(cohort+1);(r/(label+'-command.json')).write_text(json.dumps(cmd,indent=2)+'\n');run=dict(cohort=cohort,reverse=bool(cohort),argv=cmd,status='running',started_ns=time.monotonic_ns());state['runs'].append(run)
 with (r/(label+'.stdout')).open('x') as o,(r/(label+'.stderr')).open('x') as e:
  p=subprocess.Popen(cmd,stdout=o,stderr=e,env=dict(os.environ,MTL_CAPTURE_ENABLED='0'),start_new_session=True);run['pid']=p.pid;save();print('Strict serial screen',cohort,'PID',p.pid,flush=True);rc=p.wait()
 run.update(status='terminal',returncode=rc,finished_ns=time.monotonic_ns());save();print('Strict screen terminal',cohort,rc,flush=True)
 if rc:state['status']='failed';save();raise RuntimeError('Retained failed screen; inspect native logs before any next attempt')
state['status']='collected_pending_audit';save()
