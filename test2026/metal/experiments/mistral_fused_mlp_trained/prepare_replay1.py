"""Bind actual trained activations/weights to a full MLP correctness replay."""
from pathlib import Path
import hashlib,json,shutil
import numpy as np
from safetensors import safe_open
ROOT=Path(__file__).resolve().parent
M=ROOT.parent
OLD=M/'mistral-fused-gate-silu-screen1'
BASE=M/'mistral-half-attention-native1'
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
audit=json.loads((ROOT/'capture-audit1.json').read_text());assert audit['passed']
assert sha(OLD/'provenance1.json')=='e140c921b7ead252a07a820452fb90a71cdf4acec64974481521ba712901eb22'
old=json.loads((OLD/'provenance1.json').read_text())['files_sha256']
for name in ['combined1.metal','ordinary1.metal']:
    assert sha(OLD/name)==old[name];shutil.copyfile(OLD/name,ROOT/name)
assets=ROOT/'assets1';assets.mkdir()
model=Path(json.loads((BASE/'model-bindings7.json').read_text())['model'])
assert sha(model/'model.safetensors')=='d7182d886f93a9efa73d6d556ee33ffe839a4e09e15965eeef46f295ab5b503d'
with safe_open(model/'model.safetensors',framework='np') as handle:
    for layer in [0,15,31]:
        folder=assets/('layer-'+str(layer));folder.mkdir()
        for projection in ['gate','up']:
            for name,suffix in [('weight','u32'),('scales','f16'),('biases','f16')]:
                relative='assets1/'+folder.name+'/'+projection+'-'+name+'.'+suffix
                assert sha(OLD/relative)==old[relative]
                shutil.copyfile(OLD/relative,folder/(projection+'-'+name+'.'+suffix))
        for name,suffix,dtype in [('weight','u32','<u4'),('scales','f16','<f2'),('biases','f16','<f2')]:
            value=handle.get_tensor(f'model.layers.{layer}.mlp.down_proj.{name}')
            assert value.dtype==np.dtype(dtype) and value.shape==(4096,14336//(8 if name=='weight' else 64))
            value.tofile(folder/('down-'+name+'.'+suffix))
for relative,digest in audit['capture_files_sha256'].items():assert sha(ROOT/relative)==digest
context=(BASE/'source7/src/backends/metal/metal_context.mm').read_text()
command=json.loads((BASE/'configure-on5-command.json').read_text())
assert '-DGARNET_METAL_NATIVE_BFLOAT_BODY=OFF' in command
assert context.count('compileOptions=[MTLCompileOptions new];compileOptions.languageVersion=MTLLanguageVersion3_1;')==1
binding=dict(scope=__doc__,capture_audit_sha256=sha(ROOT/'capture-audit1.json'),trained_weight_layers=[0,15,31],rows=512,
    assets_sha256={str(p.relative_to(ROOT)):sha(p) for p in sorted(assets.rglob('*')) if p.is_file()},capture_files_sha256=audit['capture_files_sha256'],
    combined_sha256=sha(ROOT/'combined1.metal'),ordinary_sha256=sha(ROOT/'ordinary1.metal'),
    native_main_compile_options='nil/default: native_bfloat_body=OFF; MSL3.1 assignment is conditional on that disabled feature.',
    prior_standalone_control='Explicit MSL3.1 was a surrogate. Its numerical comparison with actual native SiLU is checked separately; its historical timings do not prove actual-default native performance.',
    control_projection='Original packed affine Safe MSL4, strict descriptor; actual native dense512 row geometry.',candidate='Fused gate/up Fast MSL4, direct half output, original Safe down GEMM.',
    correctness_gate='Exact complete native-captured gate/up/SiLU/down FP32 controls and fused post-SiLU half output; no tolerance relaxation.',
    inference_or_serving_admission=False,goal_complete=False)
(ROOT/'replay-bindings1.json').write_text(json.dumps(binding,indent=2)+'\n');print(json.dumps(dict(prepared=True,assets=len(binding['assets_sha256']),actual_native_main_options='nil/default')))
