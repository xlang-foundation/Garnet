"""Verify the package and replay every independent CPU report."""
from pathlib import Path
import hashlib,json,subprocess,sys
K=Path(__file__).resolve().parent
assert len(sys.argv)==2,'FRESH_OUTPUT_DIR'
O=Path(sys.argv[1]).resolve();assert not O.exists()
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
index=json.loads((K/'qualification1.json').read_text())
for name,h in index['files_sha256'].items():assert sha(K/name)==h,name
assert {str(p.relative_to(K)) for p in K.rglob('*') if p.is_file()}==set(index['files_sha256'])|{'qualification1.json'}
reference=json.loads((K/'source-reference.json').read_text());R=Path(reference['raw_native_root'])
assert sha(R/reference['raw_native_provenance_name'])==reference['raw_native_provenance_sha256']
assert sha(Path(reference['raw_source_replay_root'])/'provenance1.json')==reference['raw_source_replay_provenance_sha256']
subprocess.run([sys.executable,'-B',str(K/'replay2.py'),str(R),str(O)],check=True)
print('Native package: all complete CPU reports reproduced exactly.')
