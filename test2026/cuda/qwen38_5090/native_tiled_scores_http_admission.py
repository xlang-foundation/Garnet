"""Explicit new-DLL/tiled HTTP admission; old baseline proof retains its identity."""
from pathlib import Path
import hashlib,json,os,zipfile

BASELINE='1a1d0229a1c0c24a8397a4dee7a614bcedf97b3bd3293e58d752ed7de30eb498'
CANDIDATE='2533ad6867bb33137420272c975d33a26fb3637c1229fa58f5ad5dfaf1657ff7'
MODEL='bf16-scores-tiled-api-model-admission-attempt1'
MODEL_ARCHIVE='cb4effcb8021b4a8320b304646e49f70e17f4161b2447a9acf1c006231f276c2'
sha=lambda raw:hashlib.sha256(raw).hexdigest()

def validate_model(proof,workflow,library):
    assert library==proof['candidate_library_sha256']==workflow['candidate_library_sha256']==CANDIDATE
    assert proof['baseline_library_sha256']==workflow['baseline_library_sha256']==BASELINE
    assert proof['status']=='native_tiled_scores_model_replayed' and proof['archive_sha256']==MODEL_ARCHIVE
    assert proof['baseline_source_and_runtime_restored'] is True and proof['performance_admitted'] is proof['two_times_target_achieved'] is False
    assert type(workflow['exit_code']) is int and workflow['exit_code']==0 and workflow['cleanup_finished'] is workflow['baseline_library_restored'] is True
    assert workflow['source_sha256']==workflow['source_after_sha256'] and workflow['runtime_dependencies']==workflow['runtime_dependencies_after']
    assert workflow['environment_before']==workflow['environment_after']
    assert set(proof['routes'])=={'default','tiled'}
    for mode in ('default','tiled'):
        route=proof['routes'][mode]
        assert type(route['tiled_scores_enabled']) is bool and route['tiled_scores_enabled']==(mode=='tiled')
        assert type(route['captured_states']) is int and route['captured_states']==32 and len(route['cases'])==6
        assert route['complete_retained_baseline_states_and_raw_vectors_exact'] is True
        for c in route['cases']:
            assert c['actual_prompt_tokens']==3968 and c['actual_output_tokens']==128 and c['target_complete_state_and_raw_vectors_exact'] is True

def require(root,mode,library_sha256):
    if mode!='mtp':raise ValueError('Explicit tiled-score HTTP worker requires MTP mode')
    if os.environ.get('GARNET_QWEN38_BF16_LOW_BATCH_TILED_SCORES')!='1':raise ValueError('Explicit tiled native route must be selected before model loading')
    root=Path(root);proof_path=root/'results'/(MODEL+'-verified.json');proof=json.loads(proof_path.read_bytes());archive=root/'results'/(MODEL+'-evidence.zip')
    assert sha(archive.read_bytes())==proof['archive_sha256']==MODEL_ARCHIVE
    with zipfile.ZipFile(archive) as z:
        manifest=json.loads(z.read('evidence-manifest.json'));assert z.testzip() is None and len(z.namelist())==len(set(z.namelist()))==len(manifest['files'])+1
        for n,h in manifest['files'].items():assert sha(z.read(n))==h
        w=json.loads(z.read('results/'+MODEL+'/workflow.json'))
    assert w==json.loads((root/'results'/MODEL/'workflow.json').read_bytes());validate_model(proof,w,library_sha256)
    assert sha((root/'build/bin/garnet.dll').read_bytes())==library_sha256
    for n,h in w['source_after_sha256'].items():assert sha((root/n).read_bytes())==h,('Qualified tiled model source drift',n)
    runtime={**w['runtime_dependencies_after'],'garnet.dll':CANDIDATE}
    for n,h in runtime.items():assert sha((root/'build/bin'/n).read_bytes())==h
    for n,h in w['environment_after'].items():assert sha(Path(n).read_bytes())==h
    old='mtp-prefill16-full4k-admission-attempt1';old_path=root/'results'/(old+'-verified.json');q=json.loads(old_path.read_bytes())
    assert q['status']=='mtp_prefill16_full4k_target_state_replayed' and q['library_sha256']==BASELINE and q['actual_full4k_correctness_qualified'] is True and q['actual_full4k_memory_fit_observed'] is True
    assert sha((root/'results'/(old+'-evidence.zip')).read_bytes())==q['archive_sha256']==proof['baseline_archive_sha256']
    source='Garnet/xModel/qwen35/vl_0_8b/cuda_mtp_prefill16_study.py'
    return {'full4k_archive_sha256':q['archive_sha256'],'full4k_proof_sha256':sha(old_path.read_bytes()),'candidate_source_sha256':w['source_after_sha256'][source],
            'draft_prefill_chunk':16,'draft_decode_chunk':4,'extra_draft_engine_weights_duplicated':True,'http_quality_or_performance_qualified':False,
            'tiled_model_archive_sha256':proof['archive_sha256'],'tiled_model_proof_sha256':sha(proof_path.read_bytes()),'candidate_library_sha256':CANDIDATE,'baseline_library_sha256':BASELINE,
            'tiled_scores_enabled':True,'scope':'Candidate2533 default and tiled complete model states/raw vectors/IDs exactly match baseline1a1d across actual twelve4Kcases. Old baseline proof identity preserved. HTTP protocol/805quality/fixed4K/performance still require actual execution.'}
