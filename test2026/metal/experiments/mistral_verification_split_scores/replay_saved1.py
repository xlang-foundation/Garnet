"""Reproduce all six complete saved audits from frozen local evidence; no GPU."""
from pathlib import Path
import hashlib,json,subprocess,sys
HERE=Path(__file__).resolve().parent
read=lambda p:json.loads(p.read_text())
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert len(sys.argv)==2,'usage: replay_saved1.py FRESH-REPORT-DIRECTORY'
q=read(HERE/'qualification1.json')
for n,h in q['packaged_files_sha256'].items():assert sha(HERE/n)==h,n
for root,h in q['study_provenance_sha256'].items():assert sha(Path(root)/'provenance1.json')==h,root
root=Path(q['native_study']);original=root/'replay_saved2.py'
assert sha(original)==sha(HERE/'native/replay_saved2.py')
subprocess.run([sys.executable,'-B',str(original),str(Path(sys.argv[1]).resolve())],check=True)
proof=read(Path(sys.argv[1])/'replay-proof1.json')
assert proof['passed'] and len(proof['reports'])==6 and all(r['complete_report_exact'] for r in proof['reports']) and not proof['inference_executed']
print('Packaged sources and six complete saved audits verified; no inference')
