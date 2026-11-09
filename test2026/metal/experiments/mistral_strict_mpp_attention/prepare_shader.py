"""Prepare hash-verified standalone shaders, without building the dirty checkout.
Usage: python3 prepare_shader.py ADMITTED_GARNET_SOURCE FRESH_OUTPUT_DIRECTORY
"""
from pathlib import Path
import hashlib,json,subprocess,sys
source,out=[Path(x).resolve() for x in sys.argv[1:]];here=Path(__file__).resolve().parent
b=json.loads((here/'native-bindings1.json').read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
helper=here.parent/'mistral_merged_kv_attention/prepare_shader.py';assert sha(helper)==b['base_shader_prepare_helper_sha256']
assert not out.exists()
for n,h in b['files_sha256'].items():assert sha(here/n)==h,n
subprocess.run([sys.executable,'-B',str(helper),str(source),str(out)],check=True)
base=(out/'combined1.metal').read_bytes();assert hashlib.sha256(base).hexdigest()==b['base_combined_shader_sha256']
public=base+b'\n'+(here/'strict-mpp2.metal').read_bytes()+(here/'coordinate-probe1.metal').read_bytes()
register=public+(here/'strict-register2.metal').read_bytes()
diag=public+(here/'strict-register1.metal').read_bytes()+(here/'coordinate-dump1.metal').read_bytes()
for name,data,key in [('public.metal',public,'public_combined_shader_sha256'),('register.metal',register,'register_combined_shader_sha256'),('diagnostic-rejected-map.metal',diag,'diagnostic_combined_shader_sha256')]:
 assert hashlib.sha256(data).hexdigest()==b[key],name
 (out/name).write_bytes(data)
(out/'strict-bindings1.json').write_text(json.dumps(dict(passed=True,public_sha256=sha(out/'public.metal'),register_sha256=sha(out/'register.metal'),diagnostic_sha256=sha(out/'diagnostic-rejected-map.metal'),native_runtime_modified=False),indent=2)+'\n')
print('Verified standalone strict attention sources:',out)
