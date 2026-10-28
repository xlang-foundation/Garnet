"""Record observed ordinary vLLM variants without weakening token budgets."""
import hashlib,io,json,zipfile
sha=lambda raw:hashlib.sha256(raw).hexdigest()
STABLE_METADATA=('engine', 'generation_mode', 'version', 'verified_settings', 'actual_cudagraph_mode', 'context_capacity', 'image_cache_policy', 'checkpoint_revision', 'configured_engine_context_capacity', 'engine_context_reserved_slots', 'resolved_kv_cache_tokens', 'reported_max_concurrency_full4k', 'installed_config_identity_sha256', 'launcher_sha256', 'native_http_proof_sha256')


def admit_observation(proof_raw,archive_raw,metadata):
    proof=json.loads(proof_raw)
    assert proof['status']=='vllm_output_characterization_replayed' and proof['engine']=='vllm' and proof['mode']=='ordinary'
    assert proof['archive_sha256']==sha(archive_raw) and proof['replay_verifier_sha256']==proof['archived_replay_verifier_sha256']
    assert proof['performance_admitted'] is proof['two_times_target_achieved'] is False
    assert metadata['configured_engine_context_capacity']==4096 and metadata['engine_context_reserved_slots']==0 and metadata['version']=='0.32.0'
    assert len(proof['profiles'])==1;raw=proof['profiles'][0]['raw_characterization'];assert raw['status']=='reference_concurrency_output_characterization_raw_replayed'
    for name,value in (('actual_nonstream_id_requests',192),('actual_full4k_streams',64),('actual_color_streams',32),('concurrent_color_nonstreams',48),('concurrent_color_streams',24),('actual_client_overlap_waves',135)):assert type(raw[name]) is int and raw[name]==value
    assert set(raw['serial_after_comparisons'])=={'text','image','mixed-color'}
    for rows in raw['serial_after_comparisons'].values():assert len(rows)==8 and all(r['completion_ids_exact'] is True and r['text_exact'] is True for r in rows.values())
    assert raw['physical_gpu_batch_observed'] is raw['cause_or_harmlessness_proven'] is raw['broad_quality_qualified'] is raw['performance_admitted'] is raw['two_times_target_achieved'] is False
    with zipfile.ZipFile(io.BytesIO(archive_raw)) as z:
        manifest=json.loads(z.read('evidence-manifest.json'));assert z.testzip() is None and len(z.namelist())==len(set(z.namelist()))==len(manifest['files'])+1 and len(manifest['files'])==proof['files_verified']
        for name,digest in manifest['files'].items():assert sha(z.read(name))==digest
        base='results/vllm-output-characterization-attempt1/';workflow=json.loads(z.read(base+'workflow.json'));prior=json.loads(z.read(base+'metadata.json'))
    assert type(workflow['exit_code']) is int and workflow['exit_code']==0 and workflow['cleanup_finished'] is True and type(workflow['server_exit_code']) is int and workflow['source_sha256']==workflow['source_after_sha256'] and workflow['runtime_dependencies']==workflow['runtime_dependencies_after']
    for key in STABLE_METADATA:assert metadata[key]==prior[key],('Observed-output profile differs',key)
    assert metadata['engine']=='vllm' and metadata['generation_mode']=='ordinary' and metadata['source_after_startup_verified'] is metadata['owned_server_pid_verified'] is True
    return {'status':'closed_observed_output_profile_bound','policy':'record_vllm_stream_text_variants_v1','characterization_proof_sha256':sha(proof_raw),'characterization_archive_sha256':proof['archive_sha256'],'streamed_ids_observed':False,'numerical_cause_or_harmlessness_proven':False,'broad_quality_equivalence_proven':False,'scope':'Same owned ordinary vLLM/checkpoint/precision/cache/context/launcher profile as the independently replayed288request output/isolation diagnosis. Record every later stream-text variant versus actual serial ID controls; retain strict counts/usage/prompt IDs and separate semantic/quality controls. No performance, mathematical-equivalence or harmlessness admission.'}


def stream_comparison(result,control,key,body):
    assert body['ignore_eos'] is True and type(body['max_tokens']) is int and body['max_tokens']==128 and type(body['temperature']) in (int,float) and body['temperature']==0 and body['chat_template_kwargs']=={'enable_thinking':False}
    wire=json.dumps(body,separators=(',',':')).encode();assert result['key']==key and result['request_sha256']==sha(wire)
    assert result['status']=='completed' and type(result['prompt_tokens']) is int and result['prompt_tokens']==3968 and type(result['generated_tokens']) is int and result['generated_tokens']==128 and result['finish_reason']=='length'
    for name,n in (('prompt_token_ids',3968),('completion_token_ids',128)):
        ids=control[name];assert isinstance(ids,list) and len(ids)==n and all(type(t) is int and 0<=t<248320 for t in ids)
    text=result['text'];prior=control['text'];assert isinstance(text,str) and isinstance(prior,str)
    exact=text==prior;different=None if exact else next((i for i,(a,b) in enumerate(zip(text,prior)) if a!=b),min(len(text),len(prior)))
    return {'policy':'record_vllm_stream_text_variants_v1','key':key,'request_sha256':result['request_sha256'],'prompt_tokens':3968,'completion_tokens':128,'streamed_ids_observed':False,'serial_control_prompt_ids_sha256':sha(json.dumps(control['prompt_token_ids'],separators=(',',':')).encode()),'serial_control_completion_ids_sha256':sha(json.dumps(control['completion_token_ids'],separators=(',',':')).encode()),'serial_control_text_sha256':sha(prior.encode()),'stream_text_sha256':sha(text.encode()),'serial_control_text_exact':exact,'first_different_character':different,'numerical_cause_or_harmlessness_proven':False,'scope':'Actual stream text comparison only; individual streamed IDs are unavailable. Output difference is retained and not interpreted as semantic equivalence or correctness. Token/count/usage and frozen input constraints remain mandatory.'}


def replay_stream(read,key,body,control):
    if __package__:
        from .benchmark_stream import TokenEvents
    else:
        from benchmark_stream import TokenEvents
    wire=read(key+'-request.json');assert wire==json.dumps(body,separators=(',',':')).encode()
    saved=json.loads(read(key+'-result.json'));assert saved['request_sha256']==sha(wire) and saved['key']==key
    state=TokenEvents()
    for line in read(key+'-events.jsonl').splitlines():
        event=json.loads(line)
        if event['data']!='[DONE]':
            value=json.loads(event['data']);usage=value.get('usage')
            if usage is not None:
                assert type(usage.get('prompt_tokens')) is int and usage['prompt_tokens']==3968 and type(usage.get('completion_tokens')) is int and 0<=usage['completion_tokens']<=128
                assert type(usage.get('total_tokens')) is int and usage['total_tokens']==3968+usage['completion_tokens'] and not (usage.get('prompt_tokens_details') or {}).get('cached_tokens',0)
        state.feed(event['data'],event['monotonic'])
    actual=state.result(saved['submitted'],saved['completed'],128)
    for name,value in actual.items():assert saved[name]==value,('Raw stream/result differs',key,name)
    comparison=stream_comparison(saved,control,key,body);assert json.loads(read(key+'-output-comparison.json'))==comparison
    return saved,comparison
