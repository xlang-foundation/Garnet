"""Reproduce the complete qualified native source without inference."""
from pathlib import Path
import hashlib,json,shutil,subprocess,sys
PACKAGE=Path(__file__).resolve().parent;ROOT=Path(sys.argv[1]).resolve();M=ROOT.parent;K=M/'mistral-prefill-completion-policy1'
assert ROOT.parent.name=='mistral7b-local-coder' and not ROOT.exists();ROOT.mkdir()
read=lambda p:json.loads(p.read_text())
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def save(p,v):p.write_text(json.dumps(v,indent=2)+'\n')
assert sha(K/'provenance1.json')=='b3e840df717e63b7e8de3ec321bf914f22b0ebc8f3c3a19ce33bd9903e3268df'
for p in PACKAGE.iterdir():
    if p.is_file():shutil.copyfile(p,ROOT/p.name)
shutil.copytree(PACKAGE/'audit-tools3',ROOT/'audit-tools3')
tools=ROOT/'tools1';tools.mkdir()
for p in (K/'tools1').glob('*.py'):shutil.copyfile(p,tools/p.name)
base=read(K/'source-bindings1.json')['files_sha256']
for name,h in base.items():assert sha(K/'source1'/name)==h,name
S=ROOT/'source2';subprocess.run(['cp','-cR',str(K/'source1'),str(S)],check=True)
edits=read(PACKAGE/'native-edits2.json')
for name,changes in edits.items():
    p=S/name
    if not changes:
        assert name=='src/backends/metal/half_attention_profile.h' and not p.exists();p.parent.chmod(0o755);shutil.copyfile(PACKAGE/'half_attention_profile.h',p);continue
    text=p.read_text()
    for change in changes:
        assert text.count(change['old'])==1,(name,change['old'][:60]);text=text.replace(change['old'],change['new'])
    p.chmod(0o644);p.write_text(text)
files={str(p.relative_to(S)):sha(p) for p in sorted(S.rglob('*')) if p.is_file()}
canonical=hashlib.sha256(json.dumps(files,sort_keys=True,separators=(',',':')).encode()).hexdigest()
assert canonical=='81bc55abdd5e357fb02b72fb876996d56c3a444c01870830c4230781b8c004b8' and len(files)==26295,'Prepared source differs'
save(ROOT/'source-bindings2.json',dict(passed=True,source=str(S),base_source_bindings_sha256=sha(K/'source-bindings1.json'),files_sha256=files,changed_paths=sorted(edits),changes=edits,patch_sha256=sha(ROOT/'native2.patch'),shader_unchanged=True,private_half_scratch_bytes=68*1024*1024+384))
command=read(PACKAGE/'configure-template2.json')
command=[str(S) if v=='/Users/shawn/CantorAI/out/research/mistral7b-local-coder/mistral-prefill-completion-wide1/source2' else str(ROOT/'build2') if v=='/Users/shawn/CantorAI/out/research/mistral7b-local-coder/mistral-prefill-completion-wide1/build2' else v for v in command]
save(ROOT/'configure2-command.json',command);save(ROOT/'profile1.json',read(PACKAGE/'profile-template1.json'))
save(ROOT/'preparation-proof1.json',dict(passed=True,source_files=len(files),canonical_source_files_sha256=canonical,native_patch_sha256=sha(ROOT/'native2.patch'),no_compilation_or_inference=True))
print(json.dumps(dict(prepared=True,source_files=len(files),canonical_source_files_sha256=canonical)),flush=True)
