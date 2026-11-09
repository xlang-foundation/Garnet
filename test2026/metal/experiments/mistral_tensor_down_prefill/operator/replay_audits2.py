"""Fresh CPU-only report replay, preserving original paths and redirecting only report writes."""
from pathlib import Path
import hashlib,json,runpy
R=Path(__file__).resolve().parent;F=R.parent/'mistral-tensor-view-audit-replay2';assert not F.exists();F.mkdir()
reports={'audit_pilot2.py':'pilot2-audit.json','audit_pilot4.py':'pilot4-audit.json','audit_walk_source1.py':'walk-source-audit1.json','audit_timing1.py':'timing-audit1.json','audit_timing3.py':'timing-audit3.json','audit_source2.py':'source-audit2.json'}
original_write=Path.write_text;original_exists=Path.exists;active=None
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();result={}
def write(p,data,*args,**kwargs):
 if p==R/active:return original_write(F/active,data,*args,**kwargs)
 raise AssertionError('unexpected replay write '+str(p))
def exists(p):
 if p==R/active:return False
 return original_exists(p)
for script,report in reports.items():
 active=report;before=sha(R/report);Path.write_text=write;Path.exists=exists
 try:runpy.run_path(str(R/script),run_name='__main__')
 finally:Path.write_text=original_write;Path.exists=original_exists
 assert sha(R/report)==before and (R/report).read_bytes()==(F/report).read_bytes(),report
 result[report]=before
out=dict(passed=True,replayed_reports=result,no_model_or_gpu_execution=True,no_comparator_execution=True,original_reports_unchanged=True)
(F/'audit1.json').write_text(json.dumps(out,indent=2)+'\n');(R/'replay-audit2.json').write_text(json.dumps(out,indent=2)+'\n');print('Fresh replay passed',len(result),'reports')
