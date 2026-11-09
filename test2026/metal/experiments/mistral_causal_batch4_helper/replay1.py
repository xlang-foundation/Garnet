"""Replay frozen CPU audits and reconstruct source; never run inference or comparators."""
from pathlib import Path
import contextlib,hashlib,json,runpy,subprocess,sys
assert len(sys.argv)==3,'RAW_ROOT FRESH_OUTPUT_DIR'
R=Path(sys.argv[1]).resolve();P=R.parent/'mistral-causal-batch4-measure1';O=Path(sys.argv[2]).resolve();assert not O.exists();O.mkdir(parents=True)
read=lambda p:json.loads(p.read_text())
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
provenance=read(R/'provenance1.json')
for name,h in provenance['files_sha256'].items():assert sha(Path(name))==h,name
assert set(p.name for p in R.iterdir() if p.is_file())==set(provenance['root_files'])|{'provenance1.json'}
assert set(p.name for p in P.iterdir() if p.is_file())==set(provenance['measurement_root_files'])
python=Path('/Users/shawn/CantorAI/out/research/qwen-local-coder/asset-env/bin/python')
with (O/'source.stdout').open('x') as out,(O/'source.stderr').open('x') as err:
 subprocess.run([str(python),'-B',str(R/'audit_source1.py'),str(O/'source-proof')],stdout=out,stderr=err,check=True)
assert read(O/'source-proof/source-audit1.json')==read(R/'source-audit1.json')
outputs={R/'helper-audit1.json':O/'helper-audit1.json',R/'productive-audit1.json':O/'productive-audit1.json',P/'audit1.json':O/'measurement-audit1.json'}
original_write=Path.write_text;original_exists=Path.exists
def write(self,data,*a,**kw):
 destination=outputs.get(self.resolve());assert destination is not None,('unexpected auditor write',str(self));assert not original_exists(destination)
 return original_write(destination,data,*a,**kw)
def exists(self):return original_exists(outputs.get(self.resolve(),self))
for script,report in [(R/'audit_helper1.py',R/'helper-audit1.json'),(R/'audit_productive1.py',R/'productive-audit1.json'),(P/'audit1.py',P/'audit1.json')]:
 try:
  Path.write_text=write;Path.exists=exists
  with (O/(script.stem+'.stdout')).open('x') as out,contextlib.redirect_stdout(out):runpy.run_path(str(script),run_name='__main__')
 finally:Path.write_text=original_write;Path.exists=original_exists
 assert read(outputs[report])==read(report),str(report)
result=dict(passed=True,source_reconstructed=True,source_files=26561,helper_audit_exact=True,productive_full_state_audit_exact=True,measurement_audit_exact=True,no_inference=True,no_comparator_execution=True,goal_complete=False)
(O/'replay1.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
