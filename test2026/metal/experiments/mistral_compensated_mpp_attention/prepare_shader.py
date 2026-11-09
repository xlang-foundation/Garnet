"""Prepare pinned standalone source; never build or modify the dirty checkout.
Usage: python3 prepare_shader.py ADMITTED_GARNET_SOURCE FRESH_OUTPUT_DIRECTORY
"""
from pathlib import Path
import hashlib
import json
import shutil
import subprocess
import sys

source,out=[Path(x).resolve() for x in sys.argv[1:]]
here=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
b=json.loads((here/'native-bindings2.json').read_text())
helper=here.parent/'mistral_strict_mpp_attention/prepare_shader.py'
assert sha(helper)==b['base_shader_prepare_helper_sha256']
assert not out.exists()
for name,h in b['files_sha256'].items():assert sha(here/name)==h,name
subprocess.run([sys.executable,'-B',str(helper),str(source),str(out)],check=True)
base=(out/'public.metal').read_bytes()
assert hashlib.sha256(base).hexdigest()==b['base_public_shader_sha256']
candidate=base+(here/'compensated-attention2.metal').read_bytes()
assert hashlib.sha256(candidate).hexdigest()==b['candidate_combined_shader_sha256']
(out/'combined2.metal').write_bytes(candidate)
for name in b['files_sha256']:shutil.copy2(here/name,out/name)
(out/'compensated-preparation-bindings2.json').write_text(json.dumps(dict(
    passed=True,combined_shader_sha256=sha(out/'combined2.metal'),
    native_runtime_modified=False,admitted_variants=[0,1,3],
    qualified_for_serving=False),indent=2)+'\n')
print('Verified compensated attention source:',out)
