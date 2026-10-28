"""Exact separate O128 plugin using the unchanged qualified M128 context."""
from pathlib import Path
import hashlib,json
from .q128_canonical_source import sources as parent_sources
M=Path(__file__).resolve().parent;sha=lambda b:hashlib.sha256(b).hexdigest()
def sources():
    _,old,_=parent_sources();p=json.loads((M/'evidence/o128-canonical-source-derivation.json').read_bytes());assert len(old)==57 and {n:sha(b) for n,b in old.items()}==p['parent_files'];new=dict(old)
    for n in p['new_files']+p['modified_files']:new[n]=(M/'native_o128_canonical_candidate'/Path(n).name).read_bytes()
    assert len(new)==60 and {n:sha(b) for n,b in new.items()}==p['candidate_files']
    for row in p['derivations']:
        raw=old[row['source']];assert sha(raw)==row['source_sha256']
        for c in row['changes']:assert raw.count(c['old'].encode())==c['count'];raw=raw.replace(c['old'].encode(),c['new'].encode())
        assert raw==new[row['target']] and sha(raw)==row['target_sha256']
    for n,steps in p['compositions'].items():
        raw=old[n]
        for c in steps:assert raw.count(c['old'].encode())==c['count'];raw=raw.replace(c['old'].encode(),c['new'].encode())
        assert raw==new[n]
    assert all(new[n]==b for n,b in old.items() if n not in p['modified_files']);assert p['unchanged_shared_M128_context'] and p['shared_M128_workspace_bytes']==8<<20 and p['new_context_factories']==p['new_CUDA_kernels']==0 and p['old_Q_KV_GDN_plugins_wires_factories_guards_and_CUDA_kernels_unchanged'] and not p['model_or_performance_qualified'];return old,new,p
