"""Checked additive composition; this module performs no GPU/native work."""
from pathlib import Path
import hashlib,json
from .fa3_projection_witness_source import sources as parent_sources
M=Path(__file__).resolve().parent;C=M/'native_kv128_canonical_candidate'
sha=lambda b:hashlib.sha256(b).hexdigest()
def sources():
    _,old,_=parent_sources();p=json.loads((M/'evidence/kv128-canonical-source-derivation.json').read_bytes())
    assert {n:sha(b) for n,b in old.items()}==p['parent_candidate_files']
    new=dict(old)
    for name in p['new_files']+p['modified_files']:new[name]=(C/Path(name).name).read_bytes()
    assert len(old)==44 and len(new)==50 and {n:sha(b) for n,b in new.items()}==p['candidate_files']
    for row in p['derivations']:
        source=M/'native_kv128_canonical_inputs/original-modelopt_fp8_z128_canonical.h.txt' if row['source']=='src/backends/tensorrt/modelopt_fp8_z128_canonical.h' else M.parents[2]/row['source']
        raw=source.read_bytes();assert sha(raw)==row['source_sha256']
        for c in row['changes']:
            assert raw.count(c['before'].encode())==c['count'];raw=raw.replace(c['before'].encode(),c['after'].encode())
        assert raw==new['src/backends/tensorrt/'+row['target']] and sha(raw)==row['target_sha256']
    for name,steps in p['compositions'].items():
        raw=old[name]
        for c in steps:
            assert raw.count(c['before'].encode())==c['count'];raw=raw.replace(c['before'].encode(),c['after'].encode())
        assert raw==new[name]
    assert all(new[n]==b for n,b in old.items() if n not in p['modified_files'])
    assert p['opaque_constants_or_padding_guards_used'] is p['numerical_or_performance_qualified'] is False
    return old,new,p
