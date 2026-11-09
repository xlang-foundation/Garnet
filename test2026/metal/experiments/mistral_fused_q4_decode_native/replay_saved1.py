"""Reproduce complete saved audits without building or executing inference."""
from pathlib import Path
import hashlib,json,subprocess,sys
HERE=Path(__file__).resolve().parent
assert len(sys.argv)==2,'usage: replay_saved1.py FRESH-REPORT-DIRECTORY'
OUT=Path(sys.argv[1]).resolve();assert not OUT.exists();OUT.mkdir()
q=json.loads((HERE/'qualification1.json').read_text());R=Path(q['native_study'])
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
for n,h in q['packaged_native_files_sha256'].items():assert sha(HERE/'native'/n)==h,n
proof=[]
for script,report in [('audit_model2.py','model-audit2.json'),('audit_verification3.py','verification-audit3.json'),('audit_admission1.py','admission-audit1.json')]:
    assert sha(HERE/'native'/script)==sha(R/script),script
    assert sha(R/report)==q['reports_sha256'][report],report
    dest=OUT/report
    if script=='audit_admission1.py':
        # Preserve all validation statements and the recorded input root.
        # Redirect only the legacy auditor's two output path occurrences.
        code=(HERE/'native'/script).read_text();needle="ROOT/'admission-audit1.json'";assert code.count(needle)==2
        code=code.replace(needle,'Path('+repr(str(dest))+')')
        exec(compile(code,str(R/script),'exec'),{'__file__':str(R/script),'__name__':'__main__'})
    else:subprocess.run([sys.executable,'-B',str(HERE/'native'/script),str(R),str(dest)],check=True,stdout=subprocess.DEVNULL)
    actual=json.loads(dest.read_text());expected=json.loads((R/report).read_text())
    assert actual==expected,(script,'complete replay report differs')
    proof.append(dict(script=script,passed=True,saved_report_sha256=sha(R/report),replay_report_sha256=sha(dest),complete_report_exact=True))
(OUT/'replay-proof1.json').write_text(json.dumps(dict(passed=True,reports=proof,inference_executed=False),indent=2)+'\n')
print('All three full audit reports reproduced exactly; no inference executed')
