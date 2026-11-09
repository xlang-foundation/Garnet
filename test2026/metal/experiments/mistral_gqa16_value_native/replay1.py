"""Verify all raw files and reproduce full source, qualification and timing audits."""
from pathlib import Path
import hashlib,json,subprocess,sys
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
subprocess.run([sys.executable,'-B',str(R/'audit_source2.py'),str(O/'source-reconstruction')],check=True)
assert json.loads((O/'source-reconstruction/source-audit2.json').read_text())==json.loads((R/'source-audit2.json').read_text())
for auditor,report in [('audit_qualification1.py','qualification-audit1.json'),('audit_measurement1.py','measurement-audit1.json')]:
 sys.path.insert(0,str(R))
 path=R/auditor;source=path.read_text();old='R/'+repr(report);assert old in source
 source=source.replace(old,'Path('+repr(str(O/report))+')')
 exec(compile(source,str(path),'exec'),dict(__name__='__main__',__file__=str(path)))
 assert json.loads((O/report).read_text())==json.loads((R/report).read_text()),report
(O/'replay1.json').write_text(json.dumps(dict(passed=True,full_reports_equal=True,fresh_source_reconstructed=True,raw_provenance_sha256=sha(R/'provenance1.json'),no_gpu_execution=True,no_comparator_execution=True,goal_complete=False),indent=2)+'\n')
print('Complete independent CPU reports reproduced exactly.')
