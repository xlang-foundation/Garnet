"""Verify packaged sources and reproduce three complete saved audits, without inference."""
from pathlib import Path
import hashlib,json,subprocess,sys
HERE=Path(__file__).resolve().parent
read=lambda p:json.loads(p.read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert len(sys.argv)==2,'usage: replay_saved1.py FRESH-REPORT-DIRECTORY'
q=read(HERE/'qualification1.json')
for n,h in q['packaged_files_sha256'].items():assert sha(HERE/n)==h,n
for root,h in q['study_provenance_sha256'].items():assert sha(Path(root)/'provenance1.json')==h,root
script=Path(q['raw_replay_script']);assert sha(script)==sha(HERE/'raw_replay1.py')
subprocess.run([sys.executable,'-B',str(script),str(Path(sys.argv[1]).resolve())],check=True)
p=read(Path(sys.argv[1])/'replay-proof1.json');assert p['passed'] and len(p['reports'])==3 and all(x['complete_report_exact'] for x in p['reports']) and not p['inference_executed']
print('Three complete saved audits reproduced from pinned local evidence; no inference')
