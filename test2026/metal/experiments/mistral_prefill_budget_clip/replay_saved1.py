"""Reproduce three saved audit reports without inference or overwriting evidence."""
from pathlib import Path
import hashlib,json,subprocess,sys
HERE=Path(__file__).resolve().parent
assert len(sys.argv)==2,'usage: replay_saved1.py FRESH-REPORT-DIRECTORY'
OUT=Path(sys.argv[1]).resolve();assert not OUT.exists();OUT.mkdir()
q=json.loads((HERE/'qualification1.json').read_text());ROOT=Path(q['native_study']);package=HERE/'native'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
for p in package.glob('*.py'):assert sha(p)==sha(ROOT/p.name),p.name
for p in (package/'audit-tools1').glob('*.py'):assert sha(p)==sha(ROOT/'audit-tools1'/p.name),p.name
tasks=[('audit_model1.py','model-audit1.json'),('audit_admission1.py','admission-audit1.json'),('analyze_stream_windows1.py','stream-windows1.json')]
proof=[]
for script,report in tasks:
    dest=OUT/report
    if script=='audit_admission1.py':
        text=(package/script).read_text();needle="ROOT/'admission-audit1.json'";assert text.count(needle)==2
        text=text.replace(needle,'Path('+repr(str(dest))+')')
        exec(compile(text,str(ROOT/script),'exec'),{'__file__':str(ROOT/script),'__name__':'__main__'})
    else:subprocess.run([sys.executable,'-B',str(package/script),str(ROOT),str(dest)],check=True,stdout=subprocess.DEVNULL)
    assert json.loads(dest.read_text())==json.loads((ROOT/report).read_text()),script
    proof.append(dict(script=script,passed=True,saved_report_sha256=sha(ROOT/report),replay_sha256=sha(dest),reports_identical=True))
(OUT/'replay-proof1.json').write_text(json.dumps(dict(passed=True,reports=proof,new_inference_executed=False),indent=2)+'\n')
print('All three complete reports reproduced exactly; no inference')
