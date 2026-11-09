"""Reproduce complete new ownership, scheduled-model and HTTP audits without inference."""
from pathlib import Path
import hashlib,json,sys
M=Path(__file__).resolve().parent
assert len(sys.argv)==2;out=Path(sys.argv[1]).resolve();assert not out.exists();out.mkdir();proof=[]
for root,script,report in [(M/'mistral-split-score-ownership1','audit2.py','audit2.json'),(M/'mistral-scheduled-dense-verification1','audit_qualification2.py','qualification-audit2.json'),(M/'mistral-scheduled-dense-http1','audit2.py','audit2.json')]:
 original=root/script;dest=out/(root.name+'-'+report);source=original.read_text();assert source.count(repr(report))==2;source=source.replace(repr(report),repr(str(dest)));old=sys.argv;sys.argv=[str(original)]
 try:exec(compile(source,str(original),'exec'),{'__file__':str(original),'__name__':'__main__'})
 finally:sys.argv=old
 assert json.loads(dest.read_text())==json.loads((root/report).read_text()),root
 proof.append(dict(study=root.name,script=script,complete_report_exact=True,report_sha256=hashlib.sha256(dest.read_bytes()).hexdigest()))
(out/'replay-proof1.json').write_text(json.dumps(dict(passed=True,reports=proof,inference_executed=False,no_comparator_execution=True,goal_complete=False),indent=2)+'\n');print('Three complete audit reports reproduced exactly; no inference')
