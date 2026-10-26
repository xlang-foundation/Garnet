"""Admit the actual failed strict trial as diagnosis input, never as a benchmark."""
import hashlib,io,json,zipfile

PARENT='vllm-fixed4k-ordinary-attempt3'
sha=lambda raw:hashlib.sha256(raw).hexdigest()

def admit(read):
    """read accepts study-root-relative paths; validate immutable archive and proofs."""
    prefix='results/'+PARENT
    raw=read(prefix+'-evidence.zip')
    qraw=read(prefix+'-verified.json');q=json.loads(qraw)
    praw=read(prefix+'-partial-verified.json');partial=json.loads(praw)
    quality_raw=read(prefix+'-quality-verified.json');quality=json.loads(quality_raw)
    assert q['status']=='reference_fixed4k_failure_preserved' and q['engine']=='vllm' and q['mode']=='ordinary'
    assert q['archive_sha256']==partial['archive_sha256']==sha(raw)
    assert q['performance_admitted'] is q['two_times_target_achieved'] is False
    assert partial['status']=='vllm_fixed4k_attempt3_partial_controls_and_output_failure_replayed'
    assert partial['original_failure_proof_sha256']==sha(qraw)
    assert partial['actual_nonstream_4k_id_controls']==124 and partial['raw_streams_replayed']==12
    assert partial['performance_admitted'] is partial['full_4k_benchmark_complete'] is False
    variants=partial['stream_comparisons']
    assert len(variants)==12 and sum(v['text_exact'] is False for v in variants)==2
    assert all(v['individual_streamed_ids_observed'] is False and v['prompt_tokens']==3968 and v['output_tokens']==128 for v in variants)
    assert quality['status']=='adapted_mmmu_http_quality_responses_replayed' and quality['engine']=='vllm'
    assert type(quality['completed_requests']) is int and quality['completed_requests']==quality['attempted_requests']==805
    assert quality['failed_or_interrupted_cases']==[]
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        manifest=json.loads(z.read('evidence-manifest.json'))
        assert z.testzip() is None and len(z.namelist())==len(set(z.namelist()))==len(manifest['files'])+1
        for name,h in manifest['files'].items():assert sha(z.read(name))==h
        w=json.loads(z.read(prefix+'/workflow.json'))
        assert type(w['exit_code']) is int and w['exit_code']==1 and w['status']=='failed'
        assert w['cleanup_finished'] is True and type(w['server_exit_code']) is int and w['server_exit_code']==0
        assert w['source_sha256']==w['source_after_sha256'] and w['runtime_dependencies']==w['runtime_dependencies_after']
        # Quality proof was independently obtained from this archived raw run.
        qmanifest=z.read(prefix+'/client/mmmu-quality/file-manifest.json')
        assert sha(qmanifest)==quality['run_file_manifest_sha256']
    return {'job':PARENT,'archive_sha256':sha(raw),'failure_proof_sha256':sha(qraw),
            'partial_proof_sha256':sha(praw),'quality_proof_sha256':sha(quality_raw),
            'exit_code':1,'cleanup_finished':True,'performance_admitted':False}
