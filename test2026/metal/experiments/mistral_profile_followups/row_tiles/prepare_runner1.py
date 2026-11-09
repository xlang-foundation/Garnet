"""Prepare a strictly serial two-order operator timing controller."""
from pathlib import Path
R=Path(__file__).resolve().parent;s=(R/'run_pilot1.py').read_text();prefix=s.split("cmd=read(P/'host-compile3-command.json')",1)[0]
body='''cmd=read(R/'host-compile1-command.json');cmd=[str(R/'timing1.mm') if a==str(R/'pilot1.mm') else str(R/'timing1') if a==str(R/'pilot1') else a for a in cmd];run('timing-compile1',cmd)
sys.path.insert(0,str(M/'mistral-fused-q4-decode-native1/tools1'));from run_local_coder_baseline import engine_processes
assert not engine_processes(),engine_processes();assert read(R/'pilot-audit1.json')['passed'];plan=read(R/'plan1.json')
checks=read(R/'pilot1-preflight.json')['files_sha256']
for n in ['run_timing1.py','prepare_timing1.py','timing1.mm','timing1','timing-preparation1.json','timing-compile1-command.json','timing-compile1-status.json','pilot-audit1.json','audit_pilot1.py']:checks[str(R/n)]=sha(R/n)
for n,h in checks.items():assert sha(Path(n))==h,n
save('timing1-preflight.json',dict(passed=True,files_sha256=checks,collection_serial=True,model_execution=False,no_comparator_execution=True,source_or_fixture_hashes_during_timed_rounds=False))
for order in ['forward','reverse']:
 assert not engine_processes(),engine_processes()
 run('timing-'+order+'1',[str(R/'timing1'),str(R/'combined1.metal'),plan['trained_assets'],plan['trained_captures'],str(R/('timing-'+order+'1')),order]);assert read(R/('timing-'+order+'1/result1.json'))['completed']
assert not engine_processes(),engine_processes()
for n,h in checks.items():assert sha(Path(n))==h,n
save('timing1-postflight.json',dict(passed=True,checked_files=len(checks),all_owned_processes_terminal=True,no_comparator_execution=True))
'''
(R/'run_timing1.py').write_text(prefix+body);compile(prefix+body,'run_timing1.py','exec')
