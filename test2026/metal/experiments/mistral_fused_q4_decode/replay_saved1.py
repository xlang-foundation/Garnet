"""Replay all complete saved audits without GPU execution or evidence overwrite."""
from pathlib import Path
import hashlib,json,subprocess,sys
HERE=Path(__file__).resolve().parent
assert len(sys.argv)==2,'usage: replay_saved1.py FRESH-REPORT-DIRECTORY'
OUT=Path(sys.argv[1]).resolve();assert not OUT.exists();OUT.mkdir()
q=json.loads((HERE/'qualification1.json').read_text());ROOT=Path(q['study']);sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
tasks=[('audit_correctness1.py','correctness-audit1.json'),('audit_timing1.py','timing-audit1.json'),('analyze_saved_speculative_cost1.py','saved-speculative-cost1.json')];proof=[]
for script,report in tasks:
    assert sha(HERE/script)==sha(ROOT/script)==q['operator_source_sha256'][script]
    dest=OUT/report
    subprocess.run([sys.executable,'-B',str(HERE/script),str(ROOT),str(dest)],check=True,stdout=subprocess.DEVNULL)
    assert json.loads(dest.read_text())==json.loads((ROOT/report).read_text()),script
    proof.append(dict(script=script,saved_report_sha256=sha(ROOT/report),replay_sha256=sha(dest),reports_identical=True))
(OUT/'replay-proof1.json').write_text(json.dumps(dict(passed=True,reports=proof,new_gpu_execution=False),indent=2)+'\n')
print('All three complete reports reproduced exactly; no GPU execution')
