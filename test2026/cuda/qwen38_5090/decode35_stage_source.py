"""Private default-off internal operands; no numerical or performance admission."""
from pathlib import Path
import hashlib,json
from .b4_k_m4x4_source import sources as parent_sources
M=Path(__file__).resolve().parent
sha=lambda raw:hashlib.sha256(raw).hexdigest()

def sources(read_candidate=None):
    _,parent,_=parent_sources()
    p=json.loads((M/'evidence/decode35-stage-source-derivation.json').read_bytes())
    assert len(parent)==85 and {n:sha(v) for n,v in parent.items()}==p['parent_files']
    candidate=dict(parent)
    read_candidate=read_candidate or (lambda n:(M/'native_decode35_stage_candidate'/Path(n).name).read_bytes())
    for n in p['modified_files']+p['new_files']:candidate[n]=read_candidate(n)
    assert len(candidate)==86 and len(p['modified_files'])==2 and len(p['new_files'])==1
    assert {n:sha(v) for n,v in candidate.items()}==p['candidate_files']
    for row in p['derivations']:
        raw=parent[row['source']]
        for c in row['changes']:
            assert raw.count(c['old'].encode())==c['count']
            raw=raw.replace(c['old'].encode(),c['new'].encode())
        assert raw==candidate[row['target']]
    assert all(candidate[n]==v for n,v in parent.items() if n not in p['modified_files'])
    assert p['old_full_chunk_guards_and_all_CUDA_unchanged'] and p['old_counters_unchanged']
    assert p['row_words']==sum(p['stage_widths'])==106496 and len(p['stage_names'])==7
    assert not p['model_or_quality_or_performance_qualified']
    return parent,candidate,p
