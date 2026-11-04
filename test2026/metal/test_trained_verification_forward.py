"""Pin/rehash and launch one trained forward-parity engine; never a benchmark.

Args: runtime, native build, Garnet root, asset manifest, capacity, new output dir.
"""
import hashlib,json,os
from pathlib import Path
import subprocess,sys
runtime,build,source,manifest,capacity,destination=sys.argv[1:7]
output=Path(destination);output.mkdir(exist_ok=False)
manifest=Path(manifest);raw=manifest.read_bytes();assets=json.loads(raw)
assert assets['repo']=='mlx-community/Qwen3.8-27B-4bit'
assert assets['revision']=='10c35caafbb80f7dc6a7a432cdd11af10a6d4818'
assert hashlib.sha256(raw).hexdigest()=='db7aecc1a6bd8911e5737c0dc3f13089f048638669177f8c8dc331df0a6ad198'
model=Path(assets['root']);verified=[]
for entry in assets['files']:
    path=model/entry['name'];h=hashlib.sha256()
    assert path.stat().st_size==entry['bytes'],path
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''):h.update(block)
    assert h.hexdigest()==entry['sha256'],path
    verified.append(entry)
print('trained_verification_assets PASS',len(verified),flush=True)
capacity=int(capacity);assert capacity in (64,4096)
fixture=output/'fixture.json';fixture.write_text(json.dumps(dict(config=str(model/'config.json'),
    checkpoint=str(model/'model.safetensors.index.json'),capacity=capacity,output=str(output/'result.json'))))
runner=Path(__file__).with_name(sys.argv[7] if len(sys.argv)>7 else 'run_trained_verification_forward.py')
command=[runtime,'-B',str(runner),str(fixture),build,source]
paths=[Path(build)/'libgarnet.dylib',Path(runtime),Path(source)/'src/backends/metal/compiled_graph.cpp',
       Path(source)/'src/backends/metal/compiled_graph.h',Path(source)/'src/backends/metal/decoder_session.cpp',
       Path(source)/'src/tensor/model_tensor_io.h',runner,Path(__file__)]
for name in ['src/backends/metal/verification_session.cpp','src/backends/metal/decoder_session.h']:
    path=Path(source)/name
    if path.exists():paths.append(path)
provenance=dict(asset_manifest_sha256=hashlib.sha256(raw).hexdigest(),files=verified,argv=command,
    sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},benchmark=False)
(output/'provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
assert os.environ.get('GARNET_METAL_PROFILE_OPS','0')!='1'
subprocess.run(command,check=True)
