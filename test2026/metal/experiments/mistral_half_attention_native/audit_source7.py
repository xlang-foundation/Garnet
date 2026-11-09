"""Verify the clean native overlay, unchanged operators and protected checkout."""
import sys
from pathlib import Path
import hashlib,json,os,re
root=Path(sys.argv[1]).resolve() if len(sys.argv)>1 else Path(__file__).resolve().parent
workspace=root.parents[3]
base,candidate=root/'source-base1',root/'source7'
load=lambda p:json.loads(p.read_text())
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()
binding=load(root/'candidate-bindings7.json')
owned=set(binding['owned_paths'])
def manifest(directory):
    return {str(p.relative_to(directory)):('symlink:'+os.readlink(p) if p.is_symlink() else sha(p)) for p in sorted(directory.rglob('*')) if p.is_file() or p.is_symlink()}
a,b=manifest(base),manifest(candidate)
assert not set(a)-set(b),'Base files removed'
changed={name for name in b if a.get(name)!=b[name]}
assert changed==owned,(sorted(changed-owned),sorted(owned-changed))
assert {name:b[name] for name in owned}==binding['files_sha256']
assert all(a[n]==b[n] for n in a if n.startswith('xModel/'))
shader_headers=[n for n in a if 'shader' in n.lower() and n.endswith('.h')]
assert all(a[n]==b[n] for n in shader_headers)
raw=lambda p:re.findall(r'R"METAL\((.*?)\)METAL"',p.read_text(),re.S)
assert raw(base/'src/backends/metal/metal_context.mm')==raw(candidate/'src/backends/metal/metal_context.mm')
assert len(raw(candidate/'src/backends/metal/metal_context.mm'))==3
protected=load(root.parent/'mistral-fp16-attention-policy1/source-bindings1.json')['original_dirty_sha256']
assert all(sha(workspace/'Garnet'/n)==h for n,h in protected.items()),'Protected dirty checkout changed'
baseline=root.parent/'repeated-reference-comparison1/comparison.json'
assert sha(baseline)=='3ad871b986ee58880bac59771c55671af44bc7e2e56eab937e7cdba744e64566'
result=dict(passed=True,source=str(candidate),foundation=str(base),candidate_owned_files_sha256=binding['files_sha256'],
    unchanged_base_files=len(a)-len(changed&set(a)),added_files=sorted(set(b)-set(a)),
    unchanged_xmodel_files=sum(n.startswith('xModel/') for n in a),unchanged_original_shader_headers=len(shader_headers),unchanged_embedded_metal_sources=3,
    original_dirty_sha256=protected,frozen_comparison_sha256=sha(baseline),
    candidate_binding_sha256=sha(root/'candidate-bindings7.json'),
    default_option='OFF',native_integration_qualified=False,full_model_quality_qualified=False,serving_speed_claim=False,goal_complete=False)
assert 'option(GARNET_METAL_CHUNK_HALF_ATTENTION' in (candidate/'cmake/GarnetMetal.cmake').read_text()
assert 'GARNET_METAL_CHUNK_HALF_ATTENTION "' in (candidate/'cmake/GarnetMetal.cmake').read_text()
path=Path(sys.argv[2]).resolve() if len(sys.argv)>2 else root/'source-audit7.json';assert not path.exists();path.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k not in ['candidate_owned_files_sha256','original_dirty_sha256']},indent=2))
