"""Immutable separate physical B4T4 GDN FP8 canonical source composition."""
from pathlib import Path
import hashlib,json
from .b4_prefill128_source import sources as parent_sources
M=Path(__file__).resolve().parent;sha=lambda b:hashlib.sha256(b).hexdigest()

def sources(read_candidate=None):
    _,old,_=parent_sources();p=json.loads((M/'evidence/b4-decode-m16-source-derivation.json').read_bytes());assert len(old)==61 and {n:sha(b) for n,b in old.items()}==p['parent_files'];new=dict(old)
    read_candidate=read_candidate or (lambda name:(M/'native_b4_decode_m16_candidate'/Path(name).name).read_bytes())
    for n in p['modified_files']+p['new_files']:new[n]=read_candidate(n)
    assert len(new)==67 and len(p['modified_files'])==1 and len(p['new_files'])==6 and {n:sha(b) for n,b in new.items()}==p['candidate_files']
    for row in p['derivations']:
        raw=old[row['source']];assert sha(raw)==row['source_sha256']
        for c in row['changes']:assert raw.count(c['old'].encode())==c['count'];raw=raw.replace(c['old'].encode(),c['new'].encode())
        assert raw==new[row['target']] and sha(raw)==row['target_sha256']
    assert all(new[n]==b for n,b in old.items() if n not in p['modified_files'])
    assert p['new_CUDA_kernels']==0 and p['new_context_factories']==1 and p['BlasWorkspace']==4<<20 and not p['model_or_quality_or_performance_qualified']
    return old,new,p
