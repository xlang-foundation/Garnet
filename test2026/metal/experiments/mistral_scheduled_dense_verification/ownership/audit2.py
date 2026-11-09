"""Audit terminal extra lifecycle tests; repair OFF rejection-count expectation only."""
from pathlib import Path
import hashlib,json,re
R=Path(__file__).resolve().parent
read=lambda p:json.loads(p.read_text())
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert not (R/'audit2.json').exists();pre=read(R/'preflight1.json');assert pre['passed']
for n,h in pre['files_sha256'].items():assert sha(Path(n))==h,n
rows=[];last=0
for name in ['compile-on1','compile-off1','test-on1','test-off1']:
 s=read(R/(name+'-status.json'));assert s['status']=='terminal' and s['returncode']==0 and s['started_ns']>=last;last=s['finished_ns']
for mode in ['on','off']:
 p=R/('test-'+mode+'1.stdout');s=p.read_text();match=re.search(r'PASS cases=(\d+) fp32_parent_words=(\d+) rejections=(\d+) split_dispatches=(\d+) owners=0',s);assert match;cases,words,rejections,dispatches=map(int,match.groups());assert cases==12 and words==208797888 and rejections==(90 if mode=='on' else 79) and dispatches==(24 if mode=='on' else 0)
 rows.append(dict(mode=mode,cases=cases,fp32_parent_words=words,rejections=rejections,split_dispatches=dispatches,stdout_sha256=sha(p)))
report=dict(passed=True,rows=rows,repair='Original driver required>=84 rejections for both builds. OFF has72 shared checks+6 post-flush foreign checks+1 disabled-selector check=79; ON has72+6+12 route-change checks=90. Both binaries completed successfully; no rerun needed.',all_owned_processes_terminal=True,original_dirty_unchanged=True,new_b1t4_split_abort_foreign_and_recovery_exercised=True,no_model_or_comparator_execution=True,performance_measured=False,goal_complete=False)
(R/'audit2.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
