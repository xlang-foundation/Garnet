"""Explicit research worker admission for the qualified 60-source O128 candidate.

This certificate admits measurement, not production deployment or broad quality.
No compilation, native import, cache lookup or model request occurs here.
"""
from pathlib import Path
import hashlib,json,zipfile

LIBRARY='8414795d2953feae49532340b206328972c8a23767276319e6e4fb3e1912c18a'
BUNDLE='c44138f5e64b684d556830f58f09c2332d767155669c6642ba799b8445493997'
CERTIFICATES={
 'o128-full4k-model-admission-attempt1':'4058a4f1c4b542883b97c9dc046bebae440e96fb415e95f5952895ef2a9ac9d1',
 'o128-frontend-attempt1':'3cb4f2f55ca6f10c714ef4e69156133a29061434ccf0eb7cb88a3add3d82ea58',
 'o128-canonical-build-attempt1':'fd4dc3af8f7570bd53b778c77e5c2bf32f6a2db548ce71e0f20cc9572ddef06c',
}
sha=lambda b:hashlib.sha256(b).hexdigest()

def validate_closed(p,w):
    assert type(w['exit_code']) is int and w['exit_code']==0
    assert w['cleanup_finished'] is w['baseline_source_restored'] is w['baseline_library_restored'] is True
    assert w['source_before_sha256']==w['source_after_sha256']==p['files']
    assert w['installed_source_sha256']==p['installed_files']=={**p['files'],**{'Garnet/'+n:h for n,h in p['candidate_files'].items()}}
    assert len(p['candidate_files'])==60 and p['candidate_library_sha256']==w['candidate_library_sha256']==LIBRARY
    assert p['source_bundle_sha256']==BUNDLE
    assert w['runtime_dependencies']==w['runtime_dependencies_after']==p['runtime']
    assert w['environment_before']==w['environment_after']==p['environment'] and w['checkpoint_after']==p['checkpoints']
    assert w['performance_admitted'] is False

def require(root,mode,library_sha256):
    root=Path(root);assert mode in ('ordinary','mtp')
    assert library_sha256==LIBRARY==sha((root/'build/bin/garnet.dll').read_bytes())
    proofs={};archives={}
    for job,digest in CERTIFICATES.items():
        raw=(root/'results'/(job+'-verified.json')).read_bytes();assert sha(raw)==digest
        q=json.loads(raw);assert type(q['exit_code']) is int and q['exit_code']==0
        archive=root/'results'/(job+'-evidence.zip');assert sha(archive.read_bytes())==q['archive_sha256']
        proofs[job]=q;archives[job]=archive
    job='o128-full4k-model-admission-attempt1';q=proofs[job]
    assert q['full_six_case_model_qualified'] is True and q['source_library_runtime_SDK_checkpoint_restored'] is True
    assert proofs['o128-frontend-attempt1']['actual_frontend_contracts_qualified'] is True
    with zipfile.ZipFile(archives[job]) as z:
        manifest=json.loads(z.read('evidence-manifest.json'));assert z.testzip() is None
        assert len(z.namelist())==len(set(z.namelist()))==len(manifest['files'])+1
        for n,h in manifest['files'].items():assert sha(z.read(n))==h
        base='Garnet/test2026/cuda/qwen38_5090/'
        prep=z.read(base+'evidence/o128-full4k-model-preparation.json');p=json.loads(prep)
        w=json.loads(z.read('results/'+job+'/workflow.json'));assert sha(prep)==w['preparation_sha256']
    validate_closed(p,w)
    assert (root/base/'evidence/o128-full4k-model-preparation.json').read_bytes()==prep
    for n,h in p['installed_files'].items():assert sha((root/n).read_bytes())==h,('Qualified source drift',n)
    for n,h in p['runtime'].items():
        if n!='garnet.dll':assert sha((root/'build/bin'/n).read_bytes())==h,('Qualified runtime drift',n)
    for n,h in p['checkpoints'].items():assert sha((root/n).read_bytes())==h
    for n,h in p['environment'].items():assert sha(Path(n).read_bytes())==h
    assert sha((root/p['source_bundle']).read_bytes())==BUNDLE
    return dict(certificate_proof_sha256=CERTIFICATES,source_bundle_sha256=BUNDLE,
                library_sha256=LIBRARY,target_prefill_chunk=128,target_decode_chunk=4,
                draft_prefill_chunk=16,draft_decode_chunk=4,private_nonce_engines_required=True,
                quality_or_performance_qualified=False,production_deployment_qualified=False)
