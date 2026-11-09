"""Verify package/raw hashes and reproduce complete CPU reports; no GPU."""
from pathlib import Path
import hashlib,json,subprocess,sys
P=Path(__file__).resolve().parent
assert len(sys.argv)==2,'FRESH_OUTPUT_DIR'
index=json.loads((P/'qualification1.json').read_text())
for name,h in index['package_files_sha256'].items():assert hashlib.sha256((P/name).read_bytes()).hexdigest()==h,name
assert {str(p.relative_to(P)) for p in P.rglob('*') if p.is_file()}==set(index['package_files_sha256'])|{'qualification1.json'}
r=json.loads((P/'source-reference.json').read_text());R=Path(r['raw_root'])
assert hashlib.sha256((R/'provenance1.json').read_bytes()).hexdigest()==r['raw_provenance_sha256']
subprocess.run([sys.executable,'-B',str(R/'replay1.py'),str(R),sys.argv[1]],check=True)
