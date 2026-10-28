# SPDX-License-Identifier: Apache-2.0
"""CPU-only auditor check on saved cohorts plus a labelled changed-work fixture."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys

root=Path(sys.argv[1]);audit=Path(__file__).with_name('compare_qwen35_projection_workloads.py')
control=root/'matched512-garnet-2b-b1-study-activation-arena-a1.json'
candidate=root/'matched512-garnet-2b-b1-study-activation-arena-b1.json'
stem='internal-store-workload-auditor-cpu'
positive=root/(stem+'-positive.json');negative=root/(stem+'-negative.json');changed=root/(stem+'-changed-work-fixture.json')
assert not any(p.exists() for p in (positive,negative,changed)),'Preserve auditor evidence'
base=[sys.executable,str(audit),str(control)]
options=['--linear-layers','18','--activation-storage-transition']
ok=subprocess.run(base+[str(candidate),str(positive)]+options,capture_output=True,text=True)
assert ok.returncode==0 and json.loads(positive.read_text())['all_passed'],ok.stdout+ok.stderr
rows=json.loads(candidate.read_text())
for row in rows:row['diagnostic_only']=True;row['fixture_scope']='Synthetic changed-work auditor fixture, not a model run or timing result.'
rows[8]['speculative_metrics']['target_verification_calls']+=1
changed.write_text(json.dumps(rows,indent=2),encoding='utf-8')
bad=subprocess.run(base+[str(changed),str(negative)]+options,capture_output=True,text=True)
result=json.loads(negative.read_text())
assert bad.returncode!=0 and not result['all_passed']
assert result['differences']==[{'trial':rows[8]['trial'],'group':rows[8]['workload_group'],'field':'speculative_metrics'}]
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
output={'all_passed':True,'positive_exit':ok.returncode,'expected_negative_exit':bad.returncode,
        'auditor_sha256':sha(audit),'sources':{p.name:sha(p) for p in (control,candidate)},
        'changed_work_rejected':result['differences'],
        'scope':'CPU-only regression of all-request/discrete-work auditor on saved activation cohorts. Synthetic negative fixture is not benchmark data. Internal-store cohort support still requires actual new runs.'}
(root/(stem+'-summary.json')).write_text(json.dumps(output,indent=2),encoding='utf-8')
print('WORKLOAD_AUDITOR_POSITIVE_AND_CHANGED_WORK_NEGATIVE_PASS')
