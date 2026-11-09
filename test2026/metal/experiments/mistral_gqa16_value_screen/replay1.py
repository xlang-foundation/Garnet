"""Verify all frozen inputs and reproduce the independent CPU audit; no GPU."""
from pathlib import Path
import hashlib,json,sys
assert len(sys.argv)==3,'RAW_ROOT FRESH_OUTPUT_DIR'
R=Path(sys.argv[1]).resolve();O=Path(sys.argv[2]).resolve()
assert not O.exists();O.mkdir(parents=True)
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
index=json.loads((R/'provenance1.json').read_text())
for name,h in index['files_sha256'].items():assert sha(R/name)==h,name
assert {str(p.relative_to(R)) for p in R.rglob('*') if p.is_file()}==set(index['files_sha256'])|{'provenance1.json'}
path=R/'audit1.py';source=path.read_text();old="R/'audit1.json'"
assert source.count(old)==2
source=source.replace(old,'Path('+repr(str(O/'audit1.json'))+')')
exec(compile(source,str(path),'exec'),dict(__name__='__main__',__file__=str(path)))
assert json.loads((O/'audit1.json').read_text())==json.loads((R/'audit1.json').read_text())
(O/'replay1.json').write_text(json.dumps(dict(passed=True,full_report_equal=True,raw_provenance_sha256=sha(R/'provenance1.json'),no_gpu_execution=True,no_comparator_execution=True,goal_complete=False),indent=2)+'\n')
print('Independent value audit reproduced exactly.')
