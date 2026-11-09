"""Reproduce final native source bytes; no build, model load or inference."""
from pathlib import Path
import hashlib,json,shutil,subprocess,sys
HERE=Path(__file__).resolve().parent
assert len(sys.argv)==2,'usage: prepare_native1.py FRESH-STUDY-under-mistral7b-local-coder'
R=Path(sys.argv[1]).resolve();assert not R.exists();R.mkdir()
J=R.parent/'mistral-prefill-completion-wide1';S=R/'source3'
read=lambda p:json.loads(p.read_text())
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def save(p,v):p.write_text(json.dumps(v,indent=2)+'\n')
q=read(HERE/'qualification1.json')
assert sha(J/'provenance1.json')==q['base_engine_provenance_sha256']
assert sha(J/'source-bindings2.json')==q['base_source_binding_sha256']
for n,h in q['packaged_native_files_sha256'].items():assert sha(HERE/'native'/n)==h,n
subprocess.run(['cp','-cR',str(J/'source2'),str(S)],check=True)
for p in [S]+list(S.rglob('*')):
    if not p.is_symlink():p.chmod(p.stat().st_mode|0o200|(0o100 if p.is_dir() else 0))
base=read(J/'source-bindings2.json')['files_sha256']
for n,h in base.items():assert sha(S/n)==h,n
for n in q['new_native_files']:
    p=S/n;assert not p.exists();p.parent.mkdir(parents=True,exist_ok=True)
    shutil.copyfile(HERE/'native/new-files'/n,p)
skipped=0
for e in read(HERE/'native/native-edits3.json'):
    p=S/e['path'];s=p.read_text()
    if e['path']=='test2026/metal/fused_q4_decode_test.cpp':
        # The supplied new unit is already the corrected final source3 version.
        assert e['new'] in s and e['old'] not in s;skipped+=1;continue
    if 'append' in e:s+=e['append']
    else:
        assert s.count(e['old'])==e['count'],(e['path'],e['old'][:80])
        s=s.replace(e['old'],e['new'])
    p.write_text(s)
assert skipped==1
files={str(p.relative_to(S)):sha(p) for p in sorted(S.rglob('*')) if p.is_file()}
expected=dict(base);expected.update(q['native_source_changes_sha256']);assert files==expected
canonical=hashlib.sha256(json.dumps(files,sort_keys=True,separators=(',',':')).encode()).hexdigest()
assert canonical==q['native_source_content_sha256'] and len(files)==q['source_file_count']
save(R/'source-bindings3.json',dict(passed=True,source=str(S),files_sha256=files,canonical_source_sha256=canonical))
for name in ['build_native3.py','worker-model2.py','run_model2.py','audit_model2.py','worker-verification3.py','run_verification3.py','audit_verification3.py','verification-preparation3.json','run_admission1.py','audit_admission1.py','admission-preparation1.json']:
    shutil.copyfile(HERE/'native'/name,R/name)
for name in ['tools1','audit-tools1']:shutil.copytree(HERE/'native'/name,R/name)
old=q['native_study']
def relocate(v):
    if isinstance(v,str):return v.replace(old,str(R))
    if isinstance(v,list):return [relocate(x) for x in v]
    if isinstance(v,dict):return {k:relocate(x) for k,x in v.items()}
    return v
for name in ['profile-model2.json','profile-admission1.json','engine-admission1.json','admission-plan1.json']:
    save(R/name,relocate(read(HERE/'native'/name)))
save(R/'model-preparation2.json',dict(worker_sha256=sha(R/'worker-model2.py'),profile_sha256=sha(R/'profile-model2.json'),session_preparation_fixed=True,performance_measurement=False))
shutil.copyfile(HERE/'refresh_serving1.py',R/'refresh_serving1.py')
save(R/'source-replay1.json',dict(passed=True,source_files=len(files),canonical_source_sha256=canonical,inference_executed=False,build_executed=False))
print(json.dumps(dict(passed=True,source_files=len(files),canonical_source_sha256=canonical)))
