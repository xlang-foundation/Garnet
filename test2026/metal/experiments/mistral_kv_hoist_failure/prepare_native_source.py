"""Prepare isolated fault-test source from the pinned admitted hoist source.
Usage: python3 prepare_native_source.py REPOSITORY ADMITTED_SOURCE ADMITTED_MANIFEST NEW_SOURCE
"""
from pathlib import Path
import hashlib,json,subprocess,sys
repo,admitted,manifest,out=[Path(v).resolve() for v in sys.argv[1:]]
here=Path(__file__).resolve().parent
assert not out.exists()
subprocess.run([sys.executable,str(here.parent/'mistral_kv_commit_hoist/prepare_native_source.py'),str(repo),str(admitted),str(manifest),str(out)],check=True)
subprocess.run(['git','apply','--check',str(here/'test-faults1.patch')],cwd=out,check=True)
subprocess.run(['git','apply',str(here/'test-faults1.patch')],cwd=out,check=True)
a=json.loads((here/'native-bindings1.json').read_text())
for name,h in a['candidate_owned_files_sha256'].items():assert hashlib.sha256((out/name).read_bytes()).hexdigest()==h,name
(out.parent/(out.name+'-fault-preparation-audit.json')).write_text(json.dumps(dict(passed=True,candidate_owned_files_sha256=a['candidate_owned_files_sha256'],default_fault_option='OFF'),indent=2)+'\n')
print('Verified native fault-test source:',out)
