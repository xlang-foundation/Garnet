"""Verify curated bytes and independently replay scientific reports without GPU/model execution."""
from pathlib import Path
import hashlib,json,runpy,subprocess,sys
assert len(sys.argv)==3,'RAW_NATIVE_ROOT FRESH_OUTPUT'
R=Path(sys.argv[1]).resolve();P=R.parent/'mistral-tensor-view-prefill-screen1';D=Path(__file__).resolve().parent;F=Path(sys.argv[2]).resolve();assert not F.exists();F.mkdir(parents=True)
read=lambda p:json.loads(p.read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
refs=read(D/'source-reference1.json');assert sha(R/'provenance1.json')==refs['native_provenance_sha256'] and sha(P/'provenance1.json')==refs['operator_provenance_sha256']
for n,e in read(D/'package-contents1.json')['files'].items():assert sha(D/n)==e['package_sha256'] and sha(Path(e['raw']))==e['raw_sha256'],n
write_original=Path.write_text;exists_original=Path.exists;active=None;destination=None;results={}
def write(p,data,*args,**kwargs):
 if p==active:return write_original(destination,data,*args,**kwargs)
 raise AssertionError('unexpected replay write '+str(p))
def exists(p):return False if p==active else exists_original(p)
reports=[(P,'audit_pilot2.py','pilot2-audit.json'),(P,'audit_pilot4.py','pilot4-audit.json'),(P,'audit_walk_source1.py','walk-source-audit1.json'),(P,'audit_timing1.py','timing-audit1.json'),(P,'audit_timing3.py','timing-audit3.json'),(P,'audit_source2.py','source-audit2.json'),(R,'audit_qualification1.py','qualification-audit1.json'),(R,'audit_measurement1.py','measurement-audit1.json')]
for root,script,report in reports:
 active=root/report;out=F/root.name;out.mkdir(exist_ok=True);destination=out/report;before=sha(active);Path.write_text=write;Path.exists=exists
 try:runpy.run_path(str(root/script),run_name='__main__')
 finally:Path.write_text=write_original;Path.exists=exists_original
 assert sha(active)==before and active.read_bytes()==destination.read_bytes(),report
 results[root.name+'/'+report]=before
changed=read(R/'preparation1.json')['changed_files'];Q=R.parent/'mistral-causal-batch4-helper1';S=F/'patch-source1';S.mkdir()
for n in changed:
 source=Q/'source1'/n
 if source.exists():
  target=S/n;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(source.read_bytes())
subprocess.run(['/usr/bin/git','apply','--unidiff-zero','--whitespace=error',str(D/'native-review1.patch')],cwd=S,check=True)
assert all((S/n).read_bytes()==(R/'source1'/n).read_bytes() for n in changed)
result=dict(passed=True,replayed_reports=results,patch_source_edits_identical=True,no_gpu_or_model_execution=True,no_comparator_execution=True,goal_complete=False)
(F/'audit1.json').write_text(json.dumps(result,indent=2)+'\n');print('Package replay passed:',len(results),'identical reports and native patch reconstruction')
