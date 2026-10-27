"""Immutable source composition for an unqualified B4T32/M128 FP8 experiment."""
from pathlib import Path
import hashlib,json
from .o128_canonical_source import sources as parent_sources
M=Path(__file__).resolve().parent;sha=lambda b:hashlib.sha256(b).hexdigest()

def sources():
    _,old,_=parent_sources();p=json.loads((M/'evidence/b4-prefill128-source-derivation.json').read_bytes());assert len(old)==60 and {n:sha(b) for n,b in old.items()}==p['parent_files'];new=dict(old)
    for n in p['modified_files']+p['new_files']:new[n]=(M/'native_b4_prefill128_candidate'/Path(n).name).read_bytes()
    assert len(new)==61 and len(p['modified_files'])==7 and len(p['new_files'])==1 and {n:sha(b) for n,b in new.items()}==p['candidate_files']
    for n,changes in p['derivations'].items():
        raw=old[n]
        for c in changes:assert raw.count(c['old'].encode())==c['count'];raw=raw.replace(c['old'].encode(),c['new'].encode())
        assert raw==new[n]
    assert all(new[n]==b for n,b in old.items() if n not in p['modified_files']);assert p['new_CUDA_kernels']==p['new_context_factories']==0 and not p['SDK_or_model_or_MTP_or_quality_or_performance_qualified'];return old,new,p
