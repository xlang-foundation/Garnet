"""Prepare a clean, verified experimental source tree; never build the dirty checkout.
Usage: python3 prepare_native_source.py GARNET_REPOSITORY ADMITTED_SOURCE ADMITTED_MANIFEST NEW_SOURCE
"""
from pathlib import Path
import hashlib
import json
import shutil
import subprocess
import sys

repo,admitted,manifest,out=[Path(v).resolve() for v in sys.argv[1:]]
here=Path(__file__).resolve().parent
bindings=json.loads((here/'native-bindings1.json').read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
assert not out.exists()
assert sha(manifest)==bindings['admitted_manifest_sha256']
a=json.loads(manifest.read_text());assert a['passed']
out.mkdir()
p=subprocess.Popen(['git','-C',str(repo),'archive',bindings['clean_archive']],stdout=subprocess.PIPE)
result=subprocess.run(['tar','-x','-f','-','-C',str(out)],stdin=p.stdout);p.stdout.close()
assert result.returncode==0 and p.wait()==0
for name,h in a['owned_files_sha256'].items():
    source=admitted/name;assert sha(source)==h,name
    destination=out/name;destination.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,destination)
verified=0
for name,h in {**a['tracked_files_sha256'],**a['owned_files_sha256']}.items():
    if name=='CMakeLists.txt' or name.startswith(('src/','cmake/','third_party/','include/','test2026/')):
        assert sha(out/name)==h,name
        verified+=1
subprocess.run(['git','apply','--check',str(here/'native-candidate1.patch')],cwd=out,check=True)
subprocess.run(['git','apply',str(here/'native-candidate1.patch')],cwd=out,check=True)
for name,h in bindings['candidate_owned_files_sha256'].items():assert sha(out/name)==h,name
(out.parent/(out.name+'-preparation-audit.json')).write_text(json.dumps(dict(passed=True,clean_archive=bindings['clean_archive'],admitted_manifest_sha256=sha(manifest),build_relevant_base_files_verified=verified,candidate_owned_files_sha256=bindings['candidate_owned_files_sha256'],default_option='OFF'),indent=2)+'\n')
print('Verified native experimental source:',out)
