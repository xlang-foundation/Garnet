"""Prepare exact final standalone half-attention source without a dirty-tree build.
Usage: python3 prepare_shader.py ADMITTED_GARNET_SOURCE FRESH_OUTPUT
"""
from pathlib import Path
import hashlib,json,shutil,subprocess,sys
source,out=[Path(v).resolve() for v in sys.argv[1:]]
here=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
b=json.loads((here/'native-bindings3.json').read_text())
helper=here.parent/'mistral_strict_mpp_attention/prepare_shader.py'
assert sha(helper)==b['base_prepare_helper_sha256'] and not out.exists()
for name,h in b['files_sha256'].items():assert sha(here/name)==h,name
subprocess.run([sys.executable,'-B',str(helper),str(source),str(out)],check=True)
base=(out/'public.metal').read_bytes()
assert hashlib.sha256(base).hexdigest()==b['base_public_shader_sha256']
combined=base+(here/'half-policy3.metal').read_bytes()
assert hashlib.sha256(combined).hexdigest()==b['combined_shader_sha256']
(out/'combined3.metal').write_bytes(combined)
for name in b['files_sha256']:shutil.copy2(here/name,out/name)
(out/'half-preparation-bindings3.json').write_text(json.dumps(dict(passed=True,combined_shader_sha256=sha(out/'combined3.metal'),native_runtime_modified=False,quality_qualified=False),indent=2)+'\n')
print('Verified standalone half attention source:',out)
