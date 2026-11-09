"""Verify frozen raw evidence and replay the complete independent CPU audit."""
from pathlib import Path
import hashlib,json,sys
KIND='gqa'
assert len(sys.argv)==3,'RAW_ROOT FRESH_OUTPUT_DIR'
R=Path(sys.argv[1]).resolve();O=Path(sys.argv[2]).resolve();assert not O.exists();O.mkdir(parents=True)
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
index=json.loads((R/'provenance1.json').read_text())
for name,h in index['files_sha256'].items():assert sha(R/name)==h,name
assert {str(p.relative_to(R)) for p in R.rglob('*') if p.is_file()}==set(index['files_sha256'])|{'provenance1.json'}
def evaluate(name,changes):
 path=R/name;source=path.read_text()
 for old,new in changes:
  assert old in source,old;source=source.replace(old,new)
 exec(compile(source,str(path),'exec'),dict(__name__='__main__',__file__=str(path)))
if KIND=='affine':
 evaluate('audit1.py',[("R/'audit1.json'","Path("+repr(str(O/'audit1.json'))+")")])
 # Redirect only report paths; the original auditor __file__/input hashes remain.
 evaluate('audit2.py',[("runpy.run_path(str(R/'audit1.py'))",'pass'),("R/'audit1.json'","Path("+repr(str(O/'audit1.json'))+")"),("R/'audit2.json'","Path("+repr(str(O/'audit2.json'))+")")])
else:evaluate('audit2.py',[("R/'audit2.json'","Path("+repr(str(O/'audit2.json'))+")")])
for name in (['audit1.json','audit2.json'] if KIND=='affine' else ['audit2.json']):
 assert json.loads((O/name).read_text())==json.loads((R/name).read_text()),name
(O/'replay1.json').write_text(json.dumps(dict(passed=True,full_reports_equal=True,raw_provenance_sha256=sha(R/'provenance1.json'),no_gpu_execution=True,no_comparator_execution=True,goal_complete=False),indent=2)+'\n')
print('Full CPU reports reproduced exactly.')
