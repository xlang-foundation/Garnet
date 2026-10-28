"""Production load_model cache behavior; does not qualify VLM arithmetic or speed."""
STRICT=('GARNET_FP8_FULL_ATTN_M32_CANONICAL','GARNET_FP8_GDN_M32_CANONICAL','GARNET_FP8_GDN_M128_CANONICAL','GARNET_FP8_GDN_Z128_CANONICAL','GARNET_FP8_GDN_Z128_RECONSTRUCT','GARNET_FP8_FULL_ATTN_KV128_CANONICAL','GARNET_FP8_FULL_ATTN_Q128_CANONICAL','GARNET_FP8_FULL_ATTN_O128_CANONICAL')
ADDITIONAL=('GARNET_FULL_CHUNK_STAGE_WITNESS','GARNET_PARALLEL_BATCH_TOP1','GARNET_QWEN38_BF16_LOW_BATCH_TILED_SCORES','GARNET_NVFP4_DOWN_CANDIDATE')
def case(key,phase,env=None,error=None):return dict(key=key,phase=phase,environment={**{n:None for n in STRICT+ADDITIONAL},**(env or {})},expected_error=error)
def cases():
    result=[case('default-miss','miss'),case('default-hit','hit'),case('empty-alias-hit','hit',{n:'' for n in STRICT+ADDITIONAL}),case('zero-alias-hit','hit',{n:'0' for n in STRICT+ADDITIONAL})]
    for i,n in enumerate(STRICT+ADDITIONAL):
        env={n:'1'}
        if n=='GARNET_FP8_GDN_Z128_RECONSTRUCT':env['GARNET_FP8_GDN_Z128_CANONICAL']='1'
        result.extend((case('policy-'+str(i)+'-miss','miss',env),case('policy-'+str(i)+'-hit','hit',env)))
    for i,n in enumerate(STRICT):result.append(case('invalid-'+str(i),'reject',{n:'2'},n+' must be0 or1'))
    for i,v in enumerate((None,'','0')):result.append(case('uncoupled-'+str(i),'reject',{'GARNET_FP8_GDN_Z128_RECONSTRUCT':'1','GARNET_FP8_GDN_Z128_CANONICAL':v},'Z128 reconstruction requires explicit canonical Z128 policy'))
    return tuple(result+[case('return-default-miss','miss'),case('return-default-hit','hit')])
CASES=cases()
def replay_case(row,spec,library):
    assert row['status']=='actual_production_cache_case_collected' and row['spec']==spec and row['actual_environment']==spec['environment'] and row['library_sha256']==library
    assert row['actual_Garnet_load_model_called'] is True and row['runtime_status']['mode']=='compiled_xmodel' and row['runtime_status']['backend']=='tensorrt'
    status=row['runtime_status'];c=status['forbidden_path_counters']
    assert all(type(c[n]) is int for n in ('graph_cache_hits','graph_cache_misses','root_x_executions'))
    if spec['phase']=='reject':
        assert status['state']=='failed' and status['error_code']=='invalid_graph_policy' and spec['expected_error'] in status['error_message']
        assert c['graph_cache_hits']==c['graph_cache_misses']==c['root_x_executions']==0 and row['cache_before']==row['cache_after'] and row['output_values'] is None
        return dict(key=spec['key'],invalid_policy_rejected_before_cache_lookup=True,prior_cache_files_unchanged=True)
    hit=spec['phase']=='hit';assert status['ready'] is True and status['state']==('engine_cache_loaded' if hit else 'compiled_engine_ready')
    for name,want in (('graph_cache_hits',int(hit)),('graph_cache_misses',int(not hit)),('root_x_executions',int(not hit))):assert type(c[name]) is int and c[name]==want
    assert row['output_values']==[1.0,6.0,15.0,28.0] and row['forward_status']=='ok'
    key=row['cache_after']['runtime_graph.cache']['fingerprint'];assert type(key) is str and len(key)==32 and set(key)<=set('0123456789abcdef')
    return dict(key=spec['key'],cache_hit=hit,fingerprint=key,tiny_GPU_output_exact=True)
def replay(report,library):
    assert report['status']=='actual_production_cache_contracts_collected' and report['fresh_process_per_case'] is True and report['model_quality_or_performance_admitted'] is False and len(report['cases'])==41
    rows=[];baseline=None;distinct=set();last=None;previous_cache={}
    for row,spec in zip(report['cases'],CASES):
        assert row['cache_before']==previous_cache
        summary=replay_case(row,spec,library);rows.append(summary)
        if spec['phase']=='hit':assert row['cache_before']==row['cache_after']
        previous_cache=row['cache_after']
        if spec['phase']=='reject':continue
        key=summary['fingerprint']
        if baseline is None:baseline=key
        if spec['phase']=='hit':assert key==last
        elif spec['key']=='return-default-miss':assert key==baseline and key!=last
        else:assert key not in distinct
        if spec['key'] in ('empty-alias-hit','zero-alias-hit'):assert key==baseline
        distinct.add(key);last=key
    assert len(distinct)==13
    return dict(status='actual_production_graph_policy_cache_contracts_replayed',cases=rows,actual_production_calls=41,misses=14,hits=16,native_rejections=11,distinct_fingerprints=13,VLM_quality_or_performance_admitted=False)
