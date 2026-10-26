"""Exact additive witness composition over the already qualified 43-source candidate."""
from pathlib import Path
import hashlib,json
from .z128_reconstructed_plugin_source import sources as parent_sources
M=Path(__file__).resolve().parent;C=M/'native_fa3_projection_witness_candidate';sha=lambda b:hashlib.sha256(b).hexdigest()
def sources():
    _,old,_=parent_sources();p=json.loads((M/'evidence/fa3-projection-witness-source-derivation.json').read_bytes());assert {n:sha(b) for n,b in old.items()}==p['parent_candidate_files']
    new=dict(old);new[p['modified_source']]=(C/'trt_builder.cpp').read_bytes();new[p['new_header']]=(C/'qwen38_fa3_projection_witness_profile.h').read_bytes();validate(old,new,p);return old,new,p
def validate(old,new,p):
    assert len(old)==43 and len(new)==44 and {n:sha(b) for n,b in new.items()}==p['candidate_files']
    raw=old[p['modified_source']];assert sha(raw)==p['source_sha256']
    for c in p['changes']:
        a=c['before'].encode();assert raw.count(a)==c['count'];raw=raw.replace(a,c['after'].encode())
    assert raw==new[p['modified_source']] and sha(raw)==p['target_sha256']
    assert all(new[n]==b for n,b in old.items() if n!=p['modified_source'])
    assert new[p['new_header']]==(C/'qwen38_fa3_projection_witness_profile.h').read_bytes() and sha(new[p['new_header']])==p['new_header_sha256']
    assert p['ordinary_math_or_old_guards_changed'] is False and p['new_CUDA_kernels']==0
    return True
