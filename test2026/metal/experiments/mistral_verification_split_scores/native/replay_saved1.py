"""Recompute all six complete saved audits, redirecting only report destinations."""
from pathlib import Path
import hashlib,json,subprocess,sys
R=Path(__file__).resolve().parent;M=R.parent;B=M/'mistral-bounded-draft-long1';V=M/'mistral-verification-split-score-screen1'
read=lambda p:json.loads(p.read_text())
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert len(sys.argv)==2;out=Path(sys.argv[1]).resolve();assert not out.exists();out.mkdir();proof=[]
for root,script,report,direct in [(B,'audit_qualification1.py','qualification-audit1.json',True),(B,'audit_measurement1.py','measurement-audit1.json',True),(B,'audit_counters1.py','counter-audit1.json',True),(V,'audit1.py','audit1.json',False),(R,'audit_qualification1.py','qualification-audit1.json',False),(R,'audit_measurement1.py','measurement-audit1.json',False)]:
 dest=out/(root.name+'-'+report);source=root/script
 if direct:subprocess.run([sys.executable,'-B',str(source),str(root),str(dest)],check=True,stdout=subprocess.DEVNULL)
 else:
  # These helpers predate a destination argument. The two output-path literals
  # are redirected; input roots, validation, controls and source hashes stay.
  text=source.read_text();token=repr(report);assert text.count(token)==2,(script,text.count(token));text=text.replace(token,repr(str(dest)))
  old=sys.argv;sys.argv=[str(source)]
  try:exec(compile(text,str(source),'exec'),{'__file__':str(source),'__name__':'__main__'})
  finally:sys.argv=old
 assert read(dest)==read(root/report),'complete audit report changed'
 proof.append(dict(study=root.name,script=script,source_sha256=sha(source),complete_report_exact=True,report_sha256=sha(dest)))
(out/'replay-proof1.json').write_text(json.dumps(dict(passed=True,reports=proof,inference_executed=False,no_comparator_execution=True,goal_complete=False),indent=2)+'\n')
print('Six complete saved audit reports reproduced exactly; no inference')
