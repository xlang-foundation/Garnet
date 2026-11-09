"""Prepare fresh source-only diagnostic studies without builds or inference."""
from pathlib import Path
import hashlib,json,shutil,sys
HERE=Path(__file__).resolve().parent
assert len(sys.argv)==3,'usage: prepare_fresh1.py FRESH-PROFILE-STUDY FRESH-PILOT-STUDY'
H=Path(sys.argv[1]).resolve();P=Path(sys.argv[2]).resolve()
assert not H.exists() and not P.exists() and H.parent==P.parent
read=lambda p:json.loads(p.read_text())
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
q=read(HERE/'qualification1.json')
for root,h in q['source_dependencies_provenance_sha256'].items():assert sha(Path(root)/'provenance1.json')==h,root
for scope,dest in [('profile',H),('pilot',P)]:
    for n,h in q['packaged_sources_sha256'][scope].items():assert sha(HERE/scope/n)==h,n
    shutil.copytree(HERE/scope,dest)
    for n,h in q['packaged_sources_sha256'][scope].items():assert sha(dest/n)==h,n
    if scope=='pilot':
        for index in [2,3]:
            path=dest/('compile'+str(index)+'-command.json')
            cmd=[v.replace(q['pilot_study'],str(dest)) for v in read(path)]
            path.write_text(json.dumps(cmd,indent=2)+'\n')
    (dest/'fresh-preparation1.json').write_text(json.dumps(dict(passed=True,all_source_bytes_exact=True,compile_commands_only_relocated=scope=='pilot',build_executed=False,inference_executed=False),indent=2)+'\n')
print('Both source studies reproduced; no build or inference executed')
