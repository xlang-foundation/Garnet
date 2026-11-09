"""Prepare an isolated direct tensor-operand screen with unchanged native controls."""
from pathlib import Path
import hashlib,json
R=Path(__file__).resolve().parent;M=R.parent;Q=M/'mistral-causal-batch4-helper1';B=M/'mistral-tiled-half-weight-screen1'
read=lambda p:json.loads(p.read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert read(Q/'productive-audit1.json')['passed']
base=(B/'combined1.metal').read_text();native=(Q/'source1/src/backends/metal/packed_affine_shader.h').read_text().split('R"METAL(\n',1)[1].split(')METAL"',1)[0]
assert native in base,'Original packed control changed'
assert 'execution_simdgroups' not in (R/'direct1.metal').read_text()
combined=base+'\n'+(R/'direct1.metal').read_text();p=R/'combined1.metal';assert not p.exists();p.write_text(combined)
plan=dict(scope='Pilot direct device-tensor matrix operands on actual trained gate/down inputs; no native/model/serving admission.',prior_distinct='Prepared-half study retained manual M16/N32/K16 cooperative fragments; descriptor-only studies retained manual fragments and inline unpack. This uses row-major half tensor operands, M32/N32/K128 or larger tiles, two-by-two SIMD groups, and no threadgroup weight staging.',base_native=str(Q),base_library_sha256=read(Q/'build-audit1.json')['library_sha256'],original_control_source_sha256=hashlib.sha256(native.encode()).hexdigest(),base_shader_sha256=sha(B/'combined1.metal'),new_shader_sha256=sha(R/'direct1.metal'),combined_sha256=sha(p),trained_assets=str(M/'mistral-fused-mlp-trained1/assets1'),trained_captures=str(M/'mistral-fused-mlp-trained1/captures1'),layer=15,rows=512,shapes=['gate:4096->14336','down:14336->4096'],modes={'inline':'Unchanged native Q4 gather plus packed projection','manual':'Prepared row-major half, original manual K16 consumer','tensor32':'Direct tensor operands,32x32x128','tensor64m':'Direct tensor operands,64x32x128','tensor64n':'Direct tensor operands,32x64x128'},required_cost_accounting='Report consumer-only and weight-expansion+consumer costs separately; neither is model/HTTP speed.',reference='https://developer.apple.com/download/files/Metal-Performance-Primitives-Programming-Guide.pdf',no_comparator_execution=True,xmodel_operator_changes=False,goal_complete=False)
(R/'plan1.json').write_text(json.dumps(plan,indent=2)+'\n');print('Prepared unchanged packed control plus direct tensor candidates.')
