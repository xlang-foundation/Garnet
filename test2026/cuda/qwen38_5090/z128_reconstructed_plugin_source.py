"""Exact composition of the qualified parent and explicit reconstructed factory."""
from pathlib import Path
import hashlib,io,json,unittest,zipfile
from unittest.mock import patch
M=Path(__file__).resolve().parent;ROOT=M.parents[2];C=M/'native_z128_reconstructed_plugin_candidate';T='src/backends/tensorrt/'
sha=lambda raw:hashlib.sha256(raw).hexdigest()

def sources():
    p=json.loads((M/'evidence/z128-reconstructed-plugin-derivation.json').read_bytes())
    raw=(M/p['parent_source_zip']).read_bytes();assert sha(raw)==p['parent_source_zip_sha256']
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        assert z.testzip() is None and len(z.namelist())==len(set(z.namelist()))==39
        old={name:z.read(name) for name in z.namelist()}
    assert {n:sha(b) for n,b in old.items()}==p['parent_candidate_files']
    new=dict(old)
    for name in p['modified_files']+p['new_files']:new[name]=(C/Path(name).name).read_bytes()
    validate(old,new,p)
    return old,new,p

def validate(old,new,p):
    assert len(new)==43 and {n:sha(b) for n,b in new.items()}==p['candidate_files']
    assert sorted(set(new)-set(old))==p['new_files'] and sorted(n for n in old if old[n]!=new[n])==p['modified_files']
    for name,changes in p['changes'].items():
        raw=old[name]
        for row in changes:
            a=row['before'].encode();assert raw.count(a)==row['count'];raw=raw.replace(a,row['after'].encode())
        assert raw==new[name]
    for ext in ('h','cpp'):
        raw=old[T+'modelopt_fp8_z128_plugin.'+ext]
        for a,b in ((b'modelopt_fp8_z128_plugin.h',b'modelopt_fp8_z128_reconstructed_plugin.h'),(b'ModelOptFP8Z128Plugin',b'ModelOptFP8Z128ReconstructedPlugin'),(b'ModelOptFP8Z128Creator',b'ModelOptFP8Z128ReconstructedCreator'),(b'GarnetModelOptFP8Z128Canonical',b'GarnetModelOptFP8Z128Reconstructed'),(b'0x5a313238',b'0x52313238'),(b'CreateModelOptFP8Z128Context()',b'CreateModelOptFP8Z128ReconstructedContext()'),(b'"fp8_z128"',b'"fp8_z128_reconstructed"')):raw=raw.replace(a,b)
        assert new[T+'modelopt_fp8_z128_reconstructed_plugin.'+ext]==raw
    for name in ('modelopt_fp8_z128_canonical.cpp','modelopt_fp8_z128_canonical.h','modelopt_fp8_z128_reconstruct_trace.h'):
        assert new[T+name]==(M/'native_z128_reconstruct_candidate'/name).read_bytes()
    # The original family, its serializer, activation quantizer and matmul remain unchanged.
    for name in ('modelopt_fp8_z128_plugin.cpp','modelopt_fp8_z128_plugin.h','modelopt_fp8_z128_profile.h'):assert old[T+name]==new[T+name]
    a=new[T+'modelopt_fp8_z128_reconstructed_plugin.cpp'];b=old[T+'modelopt_fp8_z128_plugin.cpp']
    start=b'int ModelOptFP8Z128Plugin::enqueue';end=b'\nclass ModelOptFP8Z128Creator'
    original=b[b.index(start):b.index(end)]
    transformed=original.replace(b'ModelOptFP8Z128Plugin',b'ModelOptFP8Z128ReconstructedPlugin')
    assert transformed in a
    return True

def parent_controls():
    old,new,p=sources();prep=json.loads((M/'evidence/z128-factory-trace-native-build-preparation.json').read_bytes())
    read=Path.read_bytes;readtext=Path.read_text
    mapping={str((ROOT/name).resolve()).lower():raw for name,raw in old.items()}
    def rb(path):return mapping.get(str(path.resolve()).lower(),None) if str(path.resolve()).lower() in mapping else read(path)
    def rt(path,*args,**kwargs):
        raw=mapping.get(str(path.resolve()).lower())
        if raw is None:return readtext(path,*args,**kwargs)
        return raw.decode(kwargs.get('encoding') or (args[0] if args else 'utf8'))
    # Historical provenance assertions read their exact frozen qualified sources.
    # This is explicitly source composition, not a claim that the new route was numerically tested.
    with patch.object(Path,'read_bytes',rb),patch.object(Path,'read_text',rt):
        result=unittest.TextTestRunner(verbosity=1).run(unittest.defaultTestLoader.loadTestsFromNames(prep['test_ids']))
    assert result.wasSuccessful() and result.testsRun==86
    return result.testsRun
