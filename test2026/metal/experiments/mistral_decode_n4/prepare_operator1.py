"""Reproduce qualified standalone operator sources without inference."""
from pathlib import Path
import hashlib,json,shutil,subprocess,sys
HERE=Path(__file__).resolve().parent
assert len(sys.argv)==2,'usage: prepare_operator1.py FRESH-STUDY-under-mistral7b-local-coder'
ROOT=Path(sys.argv[1]).resolve();assert not ROOT.exists();ROOT.mkdir()
for name in ['plan1.json','interleaved1.inc','prepare1.py']:shutil.copyfile(HERE/'operator'/name,ROOT/name)
subprocess.run([sys.executable,'-B',str(ROOT/'prepare1.py')],check=True)
q=json.loads((HERE/'qualification1.json').read_text())
for p in (HERE/'operator').glob('*'):
    if p.is_file() and not(ROOT/p.name).exists():shutil.copyfile(p,ROOT/p.name)
for name,h in q['operator_reproduced_sha256'].items():assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==h,name
(ROOT/'source-replay1.json').write_text(json.dumps(dict(passed=True,files_sha256=q['operator_reproduced_sha256'],inference_executed=False),indent=2)+'\n')
print('Operator shader/host sources reproduced exactly')
