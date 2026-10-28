"""Separate K M4x4 source composition; native/API/numerical validation is pending."""
from pathlib import Path
import hashlib,json
from .b4_qk_m16_source import sources as parent_sources
M=Path(__file__).resolve().parent;sha=lambda raw:hashlib.sha256(raw).hexdigest()

def sources(read_candidate=None):
    _,parent,_=parent_sources();p=json.loads((M/'evidence/b4-k-m4x4-source-derivation.json').read_bytes())
    assert len(parent)==79 and {name:sha(raw) for name,raw in parent.items()}==p['parent_files']
    candidate=dict(parent);read_candidate=read_candidate or (lambda name:(M/'native_b4_k_m4x4_candidate'/Path(name).name).read_bytes())
    for name in p['modified_files']+p['new_files']:candidate[name]=read_candidate(name)
    assert len(candidate)==85 and len(p['modified_files'])==1 and len(p['new_files'])==6
    assert {name:sha(raw) for name,raw in candidate.items()}==p['candidate_files']
    for row in p['derivations']:
        raw=parent[row['source']];assert sha(raw)==row['source_sha256']
        for change in row['changes']:
            assert raw.count(change['old'].encode())==change['count'];raw=raw.replace(change['old'].encode(),change['new'].encode())
        assert raw==candidate[row['target']] and sha(raw)==row['target_sha256']
    assert all(candidate[n]==raw for n,raw in parent.items() if n not in p['modified_files'])
    assert p['new_CUDA_kernels']==0 and p['new_context_factories']==1 and p['BlasWorkspace']==4<<20
    assert (p['outer_M'],p['actual_M_per_call'],p['matmuls_per_enqueue'],p['output_N'],p['input_K'])==(16,4,4,1024,5120)
    assert not p['model_or_quality_or_performance_qualified'];return parent,candidate,p
