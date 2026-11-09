"""Derive serial ABBA collection from the qualified native worker controller."""
from pathlib import Path
R=Path(__file__).resolve().parent;s=(R/'run_qualification1.py').read_text();prefix=s.split("idle();build=read(R/'build-audit1.json')",1)[0]
body='''idle();assert read(R/'qualification-audit1.json')['passed'];plan=read(R/'plan1.json');libraries={'on':plan['candidate_library'],'off':plan['control_library']};sequence=[(1,'off'),(1,'on'),(2,'on'),(2,'off')]
save('measurement-plan1.json',dict(sequence=sequence,minimum_decode_ratio=1.03,no_comparator_execution=True,hashes_during_collection=False,native_b1_only=True,goal_complete=False))
checks=read(R/'qualification1-preflight.json')['files_sha256'].copy()
for n in ['run_measurement1.py','audit_measurement1.py','audit_qualification1.py','qualification-audit1.json','b1-off-result1.json','b1-on-result1.json','measurement-plan1.json']:checks[str(R/n)]=sha(R/n)
for n,h in checks.items():assert sha(Path(n))==h,n
save('measurement1-preflight.json',dict(passed=True,files_sha256=checks,collection_serial=True,hashes_during_collection=False,no_comparator_execution=True))
for cohort,mode in sequence:
 idle();name=f'cohort{cohort}-{mode}';cmd=[str(W/'out/build/macos-arm64-release/bin/xlang3'),'-B',str(R/'b1_worker1.py'),libraries[mode],str(R/'source1'),str(R/'input1.json'),str(R/(name+'-result1.json')),'bounded'];save(name+'-command1.json',cmd);start=time.time_ns()
 with (R/(name+'-stdout1.log')).open('x') as o,(R/(name+'-stderr1.log')).open('x') as e:
  p=subprocess.Popen(cmd,stdout=o,stderr=e,start_new_session=True,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',GARNET_METAL_PROFILE_OPS='0'));save(name+'-pid1.json',dict(pid=p.pid,started_ns=start));print(name,p.pid,'running',flush=True);rc=p.wait()
 save(name+'-status1.json',dict(status='terminal',returncode=rc,pid=p.pid,started_ns=start,finished_ns=time.time_ns()));print(name,rc,'terminal',flush=True)
 if rc:print((R/(name+'-stderr1.log')).read_text()[-3000:],flush=True)
 assert rc==0 and read(R/(name+'-result1.json'))['completed']
idle()
for n,h in checks.items():assert sha(Path(n))==h,n
save('measurement1-postflight.json',dict(passed=True,checked_files=len(checks),all_owned_processes_terminal=True,no_comparator_execution=True))
'''
(R/'run_measurement1.py').write_text(prefix+body);compile(prefix+body,'run_measurement1.py','exec')
