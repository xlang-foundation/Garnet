"""Reproduce C4 and compact/verification saved audits without inference."""
from pathlib import Path
import hashlib,json,sys
M=Path(__file__).resolve().parent
assert len(sys.argv)==2;out=Path(sys.argv[1]).resolve();assert not out.exists();out.mkdir();reports=[]
for root,script,report in [(M/'mistral-scheduled-c4-ownership2','audit1.py','audit1.json'),(M/'mistral-compact-verification-coexist1','audit_qualification1.py','qualification-audit1.json'),(M/'mistral-compact-verification-coexist1','audit_probe1.py','probe-audit1.json')]:
 source=root/script;dest=out/(root.name+'-'+report);text=source.read_text();assert text.count(repr(report))==2;text=text.replace(repr(report),repr(str(dest)));old=sys.argv;sys.argv=[str(source)]
 try:exec(compile(text,str(source),'exec'),{'__file__':str(source),'__name__':'__main__'})
 finally:sys.argv=old
 assert json.loads(dest.read_text())==json.loads((root/report).read_text()),root
 reports.append(dict(study=root.name,script=script,complete_report_exact=True,report_sha256=hashlib.sha256(dest.read_bytes()).hexdigest()))
(out/'replay-proof1.json').write_text(json.dumps(dict(passed=True,reports=reports,inference_executed=False,no_comparator_execution=True,goal_complete=False),indent=2)+'\n');print('Three complete saved audits reproduced exactly; no inference')
