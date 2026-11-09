"""Reproduce operator sources only; no build, GPU execution or baseline rerun."""
from pathlib import Path
import hashlib,json,shutil,subprocess,sys
HERE=Path(__file__).resolve().parent
assert len(sys.argv)==2,'usage: prepare_fresh1.py FRESH-STUDY-under-mistral7b-local-coder'
ROOT=Path(sys.argv[1]).resolve();assert not ROOT.exists();ROOT.mkdir()
q=json.loads((HERE/'qualification1.json').read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
for name,h in q['operator_source_sha256'].items():
    assert sha(HERE/name)==h,name
    if not name.endswith('.metal'):shutil.copyfile(HERE/name,ROOT/name)
with (ROOT/'prepare.stdout').open('x') as out,(ROOT/'prepare.stderr').open('x') as err:
    subprocess.run([sys.executable,'-B',str(ROOT/'prepare1.py')],check=True,stdout=out,stderr=err)
for name,h in q['operator_source_sha256'].items():assert sha(ROOT/name)==h,name
(ROOT/'source-replay1.json').write_text(json.dumps(dict(passed=True,source_files=len(q['operator_source_sha256']),all_source_bytes_exact=True,new_gpu_execution=False),indent=2)+'\n')
print('All operator source bytes reproduced; no GPU execution')
