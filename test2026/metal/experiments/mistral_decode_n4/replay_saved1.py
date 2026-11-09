"""Replay saved audits; normalize only the verified auditor-file locations."""
from pathlib import Path
import hashlib,json,subprocess,sys
HERE=Path(__file__).resolve().parent
assert len(sys.argv)==2,'usage: replay_saved1.py FRESH-REPORT-DIRECTORY'
OUT=Path(sys.argv[1]).resolve();assert not OUT.exists();OUT.mkdir()
q=json.loads((HERE/'qualification1.json').read_text());operator=Path(q['operator_study']);native=Path(q['native_study'])
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
tasks=[('operator',operator,'audit_correctness1.py','correctness-audit1.json'),('operator',operator,'audit_timing1.py','timing-audit1.json'),('operator',operator,'audit_native_c1_timing2.py','native-c1-timing-audit2.json'),('native',native,'audit_model3.py','model-audit3.json'),('native',native,'audit_admission1.py','admission-audit1.json')]
proof=[]
for scope,root,script,report in tasks:
    package=HERE/scope
    for p in package.glob('audit*.py'):assert sha(p)==sha(root/p.name),p.name
    if scope=='native':
        for p in (package/'audit-tools1').glob('*.py'):assert sha(p)==sha(root/'audit-tools1'/p.name),p.name
    dest=OUT/(scope+'-'+report)
    if script=='audit_admission1.py':
        # The recorded admission auditor predates its CLI wrapper. Redirect
        # only its two output-path occurrences, retaining its original input
        # root and every validation statement. Never overwrite the saved report.
        text=(package/script).read_text();needle="ROOT/'admission-audit1.json'";assert text.count(needle)==2
        text=text.replace(needle,'Path('+repr(str(dest))+')')
        exec(compile(text,str(root/script),'exec'),{'__file__':str(root/script),'__name__':'__main__'})
    else:subprocess.run([sys.executable,'-B',str(package/script),str(root),str(dest)],check=True,stdout=subprocess.DEVNULL)
    actual=json.loads(dest.read_text());expected=json.loads((root/report).read_text());normalized=dict(actual)
    normalized['files_sha256']={str(root/Path(name).relative_to(package)) if Path(name).is_relative_to(package) else name:h for name,h in actual['files_sha256'].items()}
    assert normalized==expected,(script,'replayed report differs beyond auditor file locations')
    proof.append(dict(script=scope+'/'+script,passed=True,saved_report_sha256=sha(root/report),raw_replay_sha256=sha(dest),auditor_file_locations_only_normalized=True))
(OUT/'replay-proof1.json').write_text(json.dumps(dict(passed=True,reports=proof,inference_executed=False),indent=2)+'\n');print('All five full audit reports reproduced; only auditor file paths normalized')
