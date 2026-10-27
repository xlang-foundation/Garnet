"""Add only production cache policy material to the frozen numeric60 candidate."""
from pathlib import Path
import hashlib,json
from .o128_canonical_source import sources as parent_sources
def sources():
    m=Path(__file__).resolve().parent;_,parent,_=parent_sources();sha=lambda b:hashlib.sha256(b).hexdigest();p=json.loads((m/'evidence/fp8-policy-cache-source-derivation.json').read_bytes());assert len(parent)==60 and {n:sha(b) for n,b in parent.items()}==p['parent_files']
    raw=(m/'native_fp8_policy_cache_candidate/compiled_model_runtime-before.cpp.txt').read_bytes();assert sha(raw)==p['original_runtime_sha256']
    for c in p['changes']:assert raw.count(c['old'].encode())==c['count'];raw=raw.replace(c['old'].encode(),c['new'].encode())
    assert raw==(m/'native_fp8_policy_cache_candidate/compiled_model_runtime.cpp').read_bytes()
    candidate={**parent,p['runtime_source']:raw,p['new_header']:(m/'native_fp8_policy_cache_candidate/graph_policy_fingerprint.h').read_bytes()};assert len(candidate)==62 and {n:sha(b) for n,b in candidate.items()}==p['candidate_files'] and all(candidate[n]==b for n,b in parent.items());return parent,candidate,p
