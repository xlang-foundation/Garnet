"""Reproduce all native source bytes without compiling or running inference."""
from pathlib import Path
import hashlib,json,shutil,subprocess,sys
HERE=Path(__file__).resolve().parent
assert len(sys.argv)==2,'usage: prepare_native1.py FRESH-STUDY-under-mistral7b-local-coder'
ROOT=Path(sys.argv[1]).resolve();assert not ROOT.exists();ROOT.mkdir();J=ROOT.parent/'mistral-prefill-completion-wide1';S=ROOT/'source1'
read=lambda p:json.loads(p.read_text())
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
q=read(HERE/'qualification1.json');assert sha(J/'provenance1.json')==q['base_engine_provenance_sha256'];assert sha(J/'source-bindings2.json')==q['base_source_binding_sha256']
subprocess.run(['cp','-cR',str(J/'source2'),str(S)],check=True)
for p in [S]+list(S.rglob('*')):
    if not p.is_symlink():p.chmod(p.stat().st_mode|0o200|(0o100 if p.is_dir() else 0))
base=read(J/'source-bindings2.json')['files_sha256']
for name,h in base.items():assert sha(S/name)==h,name
for name in q['new_native_files']:
    target=S/name;assert not target.exists();target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(HERE/'native/new-files'/name,target)
for edit in read(HERE/'native/native-edits1.json'):
    p=S/edit['path'];text=p.read_text();assert text.count(edit['old'])==edit.get('count',1),edit['path'];p.write_text(text.replace(edit['old'],edit['new']))
files={str(p.relative_to(S)):sha(p) for p in sorted(S.rglob('*')) if p.is_file()};expected=dict(base);expected.update(q['native_source_changes_sha256']);assert files==expected
canonical=hashlib.sha256(json.dumps(files,sort_keys=True,separators=(',',':')).encode()).hexdigest();assert canonical==q['native_source_content_sha256']
(ROOT/'source-bindings1.json').write_text(json.dumps(dict(passed=True,source=str(S),files_sha256=files),indent=2)+'\n')
for name in ['build_native1.py','worker-model1.py','run_model1.py','audit_model1.py','fixture-model1.json','model-preparation1.json']:
    shutil.copyfile(HERE/'native'/name,ROOT/name)
for label in ['control','clip']:
    profile=read(HERE/'native'/('profile-model-'+label+'1.json'))
    profile.update(memory_inventory_path=str(ROOT/('engine-inventory-'+label+'1.json')),diagnostic_fixture=str(ROOT/'fixture-model1.json'))
    (ROOT/('profile-model-'+label+'1.json')).write_text(json.dumps(profile,indent=2)+'\n')
for p in S.rglob('*'):
    if p.is_file() and not p.is_symlink():p.chmod(0o444)
for p in sorted([S]+[p for p in S.rglob('*') if p.is_dir()],key=lambda p:len(p.parts),reverse=True):p.chmod(0o555)
(ROOT/'source-replay1.json').write_text(json.dumps(dict(passed=True,source_files=len(files),canonical_source_sha256=canonical,inference_executed=False,build_executed=False),indent=2)+'\n')
print(json.dumps(dict(passed=True,source_files=len(files),canonical_source_sha256=canonical)))
