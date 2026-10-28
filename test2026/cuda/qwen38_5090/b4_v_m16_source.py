"""Immutable separate V-only physical B4T4 source, no production promotion."""
from pathlib import Path
import hashlib,json
from .b4_decode_m16_source import sources as parent_sources
M=Path(__file__).resolve().parent;sha=lambda raw:hashlib.sha256(raw).hexdigest()
def sources(read_candidate=None):
    _,parent,_=parent_sources();p=json.loads((M/'evidence/b4-v-m16-source-derivation.json').read_bytes());assert len(parent)==67 and {name:sha(raw) for name,raw in parent.items()}==p['parent_files'];candidate=dict(parent);read_candidate=read_candidate or (lambda name:(M/'native_b4_v_m16_candidate'/Path(name).name).read_bytes())
    for name in p['modified_files']+p['new_files']:candidate[name]=read_candidate(name)
    assert len(candidate)==73 and len(p['modified_files'])==1 and len(p['new_files'])==6 and {name:sha(raw) for name,raw in candidate.items()}==p['candidate_files']
    for row in p['derivations']:
        raw=parent[row['source']];assert sha(raw)==row['source_sha256']
        for change in row['changes']:assert raw.count(change['old'].encode())==change['count'];raw=raw.replace(change['old'].encode(),change['new'].encode())
        assert raw==candidate[row['target']] and sha(raw)==row['target_sha256']
    assert all(candidate[name]==raw for name,raw in parent.items() if name not in p['modified_files']);assert p['new_CUDA_kernels']==0 and p['new_context_factories']==1 and p['BlasWorkspace']==4<<20 and not p['model_or_quality_or_performance_qualified'];return parent,candidate,p
