"""Replay full saved profiler/pilot audits and CPU-only input admission."""
from pathlib import Path
import hashlib,json,subprocess,sys
HERE=Path(__file__).resolve().parent
assert len(sys.argv)==2,'usage: replay_saved1.py FRESH-REPORT-DIRECTORY'
OUT=Path(sys.argv[1]).resolve();assert not OUT.exists();OUT.mkdir()
read=lambda p:json.loads(p.read_text())
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
q=read(HERE/'qualification1.json');proof=[]
for scope in ['profile','pilot']:
    for n,h in q['packaged_sources_sha256'][scope].items():assert sha(HERE/scope/n)==h,n
for scope,root,script,report,key in [('profile',q['profile_study'],'audit2.py','audit2.json','profile_audit_sha256'),('pilot',q['pilot_study'],'audit1.py','audit1.json','pilot_audit_sha256')]:
    root=Path(root);dest=OUT/(scope+'-'+report)
    assert sha(HERE/scope/script)==sha(root/script) and sha(root/report)==q[key]
    subprocess.run([sys.executable,'-B',str(HERE/scope/script),str(root),str(dest)],check=True,stdout=subprocess.DEVNULL)
    assert read(dest)==read(root/report),'complete saved audit differs'
    proof.append(dict(scope=scope,passed=True,complete_report_exact=True,report_sha256=sha(dest)))
dest=OUT/'draft-long-context-admission1.json'
original=Path(q['profile_study'])/'diagnose_draft_profile1.py'
assert sha(HERE/'profile/diagnose_draft_profile1.py')==sha(original)
# This legacy helper derives its sibling inputs from __file__. Keep that
# verified original input root, redirecting only its output argument.
old_args=sys.argv;sys.argv=[str(original),str(dest)]
try:exec(compile((HERE/'profile/diagnose_draft_profile1.py').read_text(),str(original),'exec'),{'__file__':str(original),'__name__':'__main__'})
finally:sys.argv=old_args
assert read(dest)==read(Path(q['profile_study'])/'draft-long-context-admission1.json') and sha(dest)==q['draft_admission_sha256']
proof.append(dict(scope='cpu_draft_admission',passed=True,complete_report_exact=True,report_sha256=sha(dest)))
(OUT/'replay-proof1.json').write_text(json.dumps(dict(passed=True,reports=proof,inference_executed=False),indent=2)+'\n')
print('Two full audits and CPU admission report reproduced exactly; no inference')
