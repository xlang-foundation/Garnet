"""Preserve the externally overlapped sweep and prepare an unchanged clean confirmation."""
from pathlib import Path
import hashlib,json,time
R=Path(__file__).resolve().parent
p=R/'measurement1-external-overlap1.json';assert not p.exists();p.write_text(json.dumps(dict(passed=False,timing_admission=False,classification='Exploratory: another authorized research chat reported an already-running archive/hash process overlapping the first helper sweep.',reporting_thread='01a1a4e7-e2f9-7623-bf5e-82ba5fd0e3ae',reported_tool_session='46876',reported_terminal_exit_code=0,reported_gpu_model_or_numerical_work=False,interval_not_yet_bounded=True,correctness_evidence_unaffected=True,raw_data_preserved=True,action='Repeat the same five-cap forward/reverse sweep after the external process is terminal and new heavy work is held.',recorded_ns=time.time_ns()),indent=2)+'\n')
s=(R/'run_measurement1.py').read_text();s=s.replace("'measurement-plan1.json'","'measurement-plan2.json'").replace("'measurement1-preflight.json'","'measurement2-preflight.json'").replace("'measurement1-postflight.json'","'measurement2-postflight.json'")
s=s.replace("'run_measurement1.py','audit_measurement1.py'","'run_measurement2.py','audit_measurement2.py'")
s=s.replace("name=f'{order}-cap{cap}'", "name=f'clean2-{order}-cap{cap}'")
(R/'run_measurement2.py').write_text(s)
s=(R/'audit_measurement1.py').read_text();s=s.replace("'measurement1-postflight.json'","'measurement2-postflight.json'").replace("'measurement1-preflight.json'","'measurement2-preflight.json'").replace("'measurement-plan1.json'","'measurement-plan2.json'").replace("'audit_measurement1.py'","'audit_measurement2.py'").replace("name=f'{order}-cap{cap}'", "name=f'clean2-{order}-cap{cap}'").replace("R/'measurement-audit1.json'","R/'measurement-audit2.json'")
s=s.replace('report=dict(passed=True,stages=rows',"report=dict(passed=True,collection_classification='Confirmation after the reported external archive terminated; same worker/library/fixtures as the exploratory first sweep.',stages=rows")
(R/'audit_measurement2.py').write_text(s)
for n in ['run_measurement2.py','audit_measurement2.py']:compile((R/n).read_text(),n,'exec')
print('Prepared clean confirmation, preserving the complete first sweep and overlap disclosure.')
