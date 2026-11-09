"""Replay both frozen operator studies without GPU execution."""
from pathlib import Path
import hashlib,json,subprocess,sys
K=Path(__file__).resolve().parent
assert len(sys.argv)==2,'FRESH_OUTPUT_DIR'
O=Path(sys.argv[1]).resolve();assert not O.exists();O.mkdir(parents=True)
index=json.loads((K/'qualification1.json').read_text())
for name,h in index['files_sha256'].items():assert hashlib.sha256((K/name).read_bytes()).hexdigest()==h,name
assert {str(p.relative_to(K)) for p in K.rglob('*') if p.is_file()}==set(index['files_sha256'])|{'qualification1.json'}
for item in json.loads((K/'source-reference.json').read_text())['raw_roots']:
 r=Path(item['root']);assert hashlib.sha256((r/'provenance1.json').read_bytes()).hexdigest()==item['provenance_sha256']
 with (O/(item['label']+'.stdout')).open('w') as stdout:
  subprocess.run([sys.executable,'-B',str(K/item['label']/'replay1.py'),str(r),str(O/item['label'])],stdout=stdout,check=True)
(O/'replay1.json').write_text(json.dumps(dict(passed=True,full_reports_equal=True,no_gpu_execution=True,no_comparator_execution=True,goal_complete=False),indent=2)+'\n')
print('Both complete CPU audits reproduced exactly.')
