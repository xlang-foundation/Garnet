"""Exact additive private observation source, independent of mutable installed headers."""
from pathlib import Path
import hashlib,json,zipfile
M=Path(__file__).resolve().parent;sha=lambda b:hashlib.sha256(b).hexdigest()
def sources():
    p=json.loads((M/'evidence/full-chunk-stage-source-derivation.json').read_bytes());parent=M/'native_full_chunk_stage_inputs/qualified-kv128-source.zip'
    assert sha(parent.read_bytes())==p['parent_bundle_sha256']=='a0063cb10bf32144058fee174c9e0959adcf91d21bf34a9901dfe2e291f17246'
    with zipfile.ZipFile(parent) as z:
        assert z.testzip() is None and len(z.namelist())==len(set(z.namelist()))==50;old={n:z.read(n) for n in z.namelist()}
    assert {n:sha(b) for n,b in old.items()}==p['parent_files'];new=dict(old)
    for n in p['modified_files']+[p['new_header']]:new[n]=(M/'native_full_chunk_stage_candidate'/Path(n).name).read_bytes()
    assert len(new)==51 and {n:sha(b) for n,b in new.items()}==p['candidate_files']
    for n,steps in p['derivations'].items():
        raw=old[n]
        for c in steps:assert raw.count(c['old'].encode())==c['count'];raw=raw.replace(c['old'].encode(),c['new'].encode())
        assert raw==new[n]
    assert all(new[n]==b for n,b in old.items() if n not in p['modified_files'])
    assert p['old_diagnostic_and_INT4_projection_guards_unchanged'] is p['original_numerical_operations_unchanged'] is True
    assert p['new_CUDA_kernels']==0 and p['model_or_performance_qualified'] is False
    return old,new,p
