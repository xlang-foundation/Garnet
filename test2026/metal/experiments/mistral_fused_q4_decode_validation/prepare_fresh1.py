"""Prepare fresh prefix and learned-draft studies; no build or inference."""
from pathlib import Path
import hashlib,json,shutil,subprocess,sys
HERE=Path(__file__).resolve().parent
assert len(sys.argv)==3,'usage: prepare_fresh1.py FRESH-PREFIX-STUDY FRESH-CODER-STUDY'
P=Path(sys.argv[1]).resolve();S=Path(sys.argv[2]).resolve()
assert not P.exists() and not S.exists() and P.parent==S.parent
read=lambda p:json.loads(p.read_text())
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
q=read(HERE/'qualification1.json')
for scope in ['prefix','coder']:
    for n,h in q['packaged_sources_sha256'][scope].items():assert sha(HERE/scope/n)==h,n
for root,expected in q['base_provenances_sha256'].items():assert sha(Path(root)/'provenance1.json')==expected,root
P.mkdir();shutil.copyfile(HERE/'prefix/prepare1.py',P/'prepare1.py')
subprocess.run([sys.executable,'-B',str(P/'prepare1.py')],check=True)
assert sha(P/'tools1/local_coder_batch_worker.py')==q['prefix_worker_sha256']
for n,h in q['packaged_sources_sha256']['prefix'].items():assert sha(P/n)==h,n
# Clarify that the setup failure in the inherited preparation history happened
# in the original study, rather than in this successful fresh preparation.
prep=read(P/'preparation1.json');prep['inherited_preparation_history_from']=q['prefix_study']
(P/'preparation1.json').write_text(json.dumps(prep,indent=2)+'\n')
shutil.copytree(HERE/'coder',S)
plan=read(S/'plan1.json');plan['input']=str(S/'input1.json');plan['prefix_gate']=str(P/'lifecycle-audit1.json')
(S/'plan1.json').write_text(json.dumps(plan,indent=2)+'\n')
assert sha(S/'worker1.py')==q['coder_worker_sha256'] and sha(S/'input1.json')==q['input_sha256']
for root in [P,S]:
    (root/'fresh-preparation1.json').write_text(json.dumps(dict(passed=True,build_executed=False,inference_executed=False,source_package=str(HERE)),indent=2)+'\n')
print('Fresh workers and fixture reproduce; no native build or inference executed')
