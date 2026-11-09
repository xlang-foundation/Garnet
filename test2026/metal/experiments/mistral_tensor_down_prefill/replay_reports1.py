"""Reproduce native qualification and timing reports in a fresh CPU-only output directory."""
from pathlib import Path
import hashlib,json,runpy
R=Path(__file__).resolve().parent;F=R.parent/'mistral-down-tensor-audit-replay1';assert not F.exists();F.mkdir()
reports={'audit_qualification1.py':'qualification-audit1.json','audit_measurement1.py':'measurement-audit1.json'}
write_original=Path.write_text;exists_original=Path.exists;active=None;results={}
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,data,*args,**kwargs):
 if p==R/active:return write_original(F/active,data,*args,**kwargs)
 raise AssertionError('unexpected replay write '+str(p))
def exists(p):return False if p==R/active else exists_original(p)
for script,report in reports.items():
 active=report;before=sha(R/report);Path.write_text=write;Path.exists=exists
 try:runpy.run_path(str(R/script),run_name='__main__')
 finally:Path.write_text=write_original;Path.exists=exists_original
 assert sha(R/report)==before and (R/report).read_bytes()==(F/report).read_bytes(),report
 results[report]=before
out=dict(passed=True,replayed_reports=results,no_model_or_gpu_execution=True,no_comparator_execution=True,original_reports_unchanged=True)
(F/'audit1.json').write_text(json.dumps(out,indent=2)+'\n');(R/'replay-audit1.json').write_text(json.dumps(out,indent=2)+'\n');print('Fresh native report replay passed')
