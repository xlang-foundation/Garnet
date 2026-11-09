"""Reproduce complete prefix and current-ordinary learned-draft audit reports."""
from pathlib import Path
import hashlib,json,subprocess,sys
HERE=Path(__file__).resolve().parent
assert len(sys.argv)==2,'usage: replay_saved1.py FRESH-REPORT-DIRECTORY'
OUT=Path(sys.argv[1]).resolve();assert not OUT.exists();OUT.mkdir()
read=lambda p:json.loads(p.read_text())
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
q=read(HERE/'qualification1.json');proof=[]
for scope,root,script,report in [('prefix',q['prefix_study'],'audit_lifecycle1.py','lifecycle-audit1.json'),('coder',q['coder_study'],'audit3.py','audit3.json')]:
    root=Path(root)
    for n,h in q['packaged_sources_sha256'][scope].items():assert sha(HERE/scope/n)==h,n
    assert sha(HERE/scope/script)==sha(root/script)
    dest=OUT/report;subprocess.run([sys.executable,'-B',str(HERE/scope/script),str(root),str(dest)],check=True,stdout=subprocess.DEVNULL)
    assert read(dest)==read(root/report),'complete audit report differs'
    assert sha(root/report)==q['reports_sha256'][scope]
    proof.append(dict(scope=scope,passed=True,complete_report_exact=True,saved_report_sha256=sha(root/report),replay_report_sha256=sha(dest)))
(OUT/'replay-proof1.json').write_text(json.dumps(dict(passed=True,reports=proof,inference_executed=False),indent=2)+'\n')
print('Both full audit reports reproduced exactly; no inference executed')
