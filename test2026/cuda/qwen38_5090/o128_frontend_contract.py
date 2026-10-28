"""Real decoder-layer frontend cases; native execution evidence is required separately."""
BASE_FLAGS={
 'GARNET_FP8_FULL_ATTN_M32_CANONICAL':'1','GARNET_FP8_GDN_M32_CANONICAL':'1',
 'GARNET_FP8_GDN_M128_CANONICAL':'1','GARNET_FP8_GDN_Z128_CANONICAL':'1',
 'GARNET_FP8_GDN_Z128_RECONSTRUCT':'1','GARNET_FP8_FULL_ATTN_KV128_CANONICAL':'1',
 'GARNET_FP8_FULL_ATTN_Q128_CANONICAL':'1','GARNET_FP8_FULL_ATTN_O128_CANONICAL':'1',
 'GARNET_FULL_CHUNK_STAGE_WITNESS':'0','GARNET_PARALLEL_BATCH_TOP1':'1',
 'GARNET_QWEN38_BF16_LOW_BATCH_TILED_SCORES':'0','GARNET_NVFP4_DOWN_CANDIDATE':'0'}
Q='native_fp8_q128_canonical_projections';O='native_fp8_o128_canonical_projections';KV='native_fp8_kv128_canonical_projections'
ZERO=('native_fp8_m128_canonical_projections','native_fp8_z128_canonical_projections','native_fp8_z128_reconstructed_projections','native_full_chunk_stage_witnesses')
def case(key,*,batch=1,tokens=128,hidden=5120,layer=3,flags=None,error=None):
    return dict(key=key,batch=batch,tokens=tokens,hidden=hidden,layer=layer,flags={**BASE_FLAGS,**(flags or {})},expected_error=error)
CASES=(
 case('enabled128'),case('disabled128',flags={n:'0' for n in ('GARNET_FP8_FULL_ATTN_Q128_CANONICAL','GARNET_FP8_FULL_ATTN_O128_CANONICAL','GARNET_FP8_FULL_ATTN_KV128_CANONICAL')}),
 case('enabled32',tokens=32),case('enabled4',tokens=4),case('nonmatching_batch128',batch=2),case('other_text_layer128',layer=7),
 *(case(key,flags={flag:value},error=flag+' must be0 or1') for key,flag,value in (
 ('invalid_Q_true','GARNET_FP8_FULL_ATTN_Q128_CANONICAL','true'),('invalid_Q_2','GARNET_FP8_FULL_ATTN_Q128_CANONICAL','2'),
 ('invalid_O_true','GARNET_FP8_FULL_ATTN_O128_CANONICAL','true'),('invalid_O_2','GARNET_FP8_FULL_ATTN_O128_CANONICAL','2'),
 ('invalid_KV','GARNET_FP8_FULL_ATTN_KV128_CANONICAL','2'),('invalid_M','GARNET_FP8_GDN_M128_CANONICAL','2'),
 ('invalid_Z','GARNET_FP8_GDN_Z128_CANONICAL','2'),('invalid_reconstruct','GARNET_FP8_GDN_Z128_RECONSTRUCT','2'))),
 case('reconstruction_without_canonical',flags={'GARNET_FP8_GDN_Z128_CANONICAL':'0'},error='Z128 reconstruction requires explicit canonical Z128 policy'),
 case('chunk129',tokens=129,error='Qwen3.5 full chunk dimensions invalid'),
 case('missing_full_attention_layer',layer=1,error='Qwen3.5 missing checkpoint weight'),
)
def expected_counts(spec):
    selected=spec['batch']==1 and spec['tokens']==128
    return {field:scale if selected and spec['flags'][flag]=='1' else 0 for field,scale,flag in ((Q,1,'GARNET_FP8_FULL_ATTN_Q128_CANONICAL'),(O,1,'GARNET_FP8_FULL_ATTN_O128_CANONICAL'),(KV,2,'GARNET_FP8_FULL_ATTN_KV128_CANONICAL'))}
def replay_case(row,spec):
    assert row['spec']==spec and row['actual_flags']==spec['flags'] and row['actual_Garnet_frontend_requested'] is True
    assert row['library_sha256']=='8414795d2953feae49532340b206328972c8a23767276319e6e4fb3e1912c18a'
    if spec['expected_error'] is not None:
        assert row['status']=='expected_native_rejection' and row['compiled'] is False and row['graph_stats'] is None
        assert type(row['native_error']) is str and spec['expected_error'] in row['native_error']
        return dict(key=spec['key'],native_rejection_verified=True)
    assert row['status']=='actual_frontend_compiled' and row['compiled'] is True and row['native_error'] is None
    graph=row['graph_stats'];counts=expected_counts(spec)
    for field,n in {**counts,**{n:0 for n in ZERO}}.items():assert type(graph[field]) is int and graph[field]==n,(spec['key'],field,graph.get(field),n)
    assert graph['checkpoint_weight_precision']=='modelopt_mixed' and graph['native_fp8_projection_weights']==4
    return dict(key=spec['key'],actual_compiled_route_counts=counts)
def replay(report):
    assert report['status']=='actual_O128_frontend_contracts_collected' and report['fresh_process_per_case'] is True and report['model_or_performance_qualified'] is False
    assert len(report['cases'])==len(CASES)
    rows=[replay_case(row,spec) for row,spec in zip(report['cases'],CASES)]
    return dict(status='actual_O128_frontend_contracts_replayed',cases=rows,compiled_cases=sum(s['expected_error'] is None for s in CASES),native_rejections=sum(s['expected_error'] is not None for s in CASES),full_model_or_quality_or_performance_admitted=False)
