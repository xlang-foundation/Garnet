"""Check selected CUDA operation coverage against decode-only MTP counters."""
from pathlib import Path
import hashlib,json,sqlite3,sys,zipfile

root,destination=map(Path,sys.argv[1:3]);name='kv-delta-decode-cuda-trace-attempt1'
proof_path=root/(name+'-verified.json');proof=json.loads(proof_path.read_bytes())
sha=lambda b:hashlib.sha256(b).hexdigest()
assert proof['status']=='compact_decode_cuda_trace_replayed' and proof['cases']==8 and proof['raw_hidden_all_exact']
assert sha((root/(name+'-evidence.zip')).read_bytes())==proof['archive_sha256']
sql=Path(sys.argv[3])
with zipfile.ZipFile(root/(name+'-evidence.zip')) as archive:
    manifest=json.loads(archive.read('evidence-manifest.json'))
    assert sha(sql.read_bytes())==manifest['files']['results/'+name+'/startup-trace.sqlite']
selectors={
 'nvfp4_matmul':lambda n:n.startswith('cutlass3x_sm120_bstensorop_'),
 'nvfp4_quantizer':lambda n:'::QuantizeActivation(' in n,
 'fp8_matmul':lambda n:n.startswith('nvjet_sm120_') or ('sm89_xmma_gemm_e4m3' in n and '_execute_split_k_kernel_' not in n),
 'fp8_split_k_followup':lambda n:'sm89_xmma_gemm_e4m3' in n and '_execute_split_k_kernel_' in n,
 'fp8_quantizer':lambda n:'::Quantize(' in n,
 'bf16_small_tile':lambda n:'cutlass_80_wmma_tensorop_bf16_s161616gemm_bf16_16x16_128x2_tn_align8' in n,
 'bf16_larger_tile':lambda n:'cutlass_80_wmma_tensorop_bf16_s161616gemm_bf16_32x32_128x2_tn_align8' in n,
 'gdn_recurrence':lambda n:'::recurrence(' in n,
 'short_context_attention':lambda n:'::compactCached<(bool)0>(' in n,
 'canonical_linear_commit':lambda n:'::copyLinear(' in n and 'Qwen35KVDelta' in n,
 'canonical_kv_scatter':lambda n:'::scatterKV(' in n and 'Qwen35KVDelta' in n,
}
windows=[]
for window in proof['windows']:
    after=window['speculative_metrics'];before=window['metrics_before_generate']
    assert before['target_verification_calls']==before['target_prefix_recomputes']==before['target_prefix_replays']==0
    assert after['mtp_prefill_calls']==before['mtp_prefill_calls']>0 and before['mtp_head_calls']==1
    target=after['target_forward_calls'];prefix=after['target_prefix_replays']
    mtp=sum(after[k]-before[k] for k in ('mtp_draft_calls','mtp_correction_calls'))
    head=after['target_head_evaluations']+after['mtp_head_calls']-before['mtp_head_calls']
    assert target==after['target_verification_calls'] and after['target_prefix_recomputes']==0
    # Diagnose the initial one-kernel-per-projection assumption from actual
    # ordered records: each matched GEMM has an additional split-K follow-up.
    connection=sqlite3.connect('file:'+sql.as_posix()+'?mode=ro',uri=True);connection.row_factory=sqlite3.Row
    try:
        paired=[dict(r) for r in connection.execute("select k.start,k.end,k.graphId,k.graphNodeId,k.streamId,k.deviceId,k.globalPid,k.correlationId,k.gridX,k.gridY,k.gridZ,s.value as name from CUPTI_ACTIVITY_KIND_KERNEL k join StringIds s on k.demangledName=s.id where k.start>=? and k.end<=? and s.value like '%sm89_xmma_gemm_e4m3%tilesize64x64x64%' order by k.start",window['marker_interval_ns'])]
    finally:connection.close()
    assert len(paired)==64*target
    for primary,followup in zip(paired[::2],paired[1::2]):
        assert '_execute_split_k_kernel_' not in primary['name'] and '_execute_split_k_kernel_' in followup['name']
        assert followup['name'].replace('_execute_split_k_kernel_','_execute_kernel_')==primary['name']
        assert primary['end']<=followup['start']
        assert all(primary[k]==followup[k] for k in ('graphId','streamId','deviceId','globalPid','correlationId'))
    expected={'nvfp4_matmul':128*target+head,'nvfp4_quantizer':128*target+head,
        'fp8_matmul':208*target,'fp8_split_k_followup':32*target,'fp8_quantizer':208*target,
        'bf16_small_tile':96*target+6*mtp,'bf16_larger_tile':2*mtp,
        'gdn_recurrence':48*(target+prefix),'short_context_attention':16*target,
        'canonical_linear_commit':target,'canonical_kv_scatter':target}
    actual={key:sum(row['count'] for row in window['kernels_by_name'] if match(row['name'])) for key,match in selectors.items()}
    assert actual==expected,(window['call'],actual,expected)
    windows.append({'call':window['call'],'target_calls':target,'prefix_replays':prefix,'mtp_body_calls_excluding_prefill':mtp,
        'head_calls_excluding_prefill':head,'actual':actual,'expected':expected,'split_k_pairs_checked':len(paired)//2,
        'split_k_grid_shapes':sorted({tuple(r[k] for k in ('gridX','gridY','gridZ')) for r in paired})})
result={'status':'selected_decode_operation_counts_matched','archive_sha256':proof['archive_sha256'],
    'proof_sha256':sha(proof_path.read_bytes()),'auditor_sha256':sha(Path(__file__).read_bytes()),'windows':windows,
    'two_x_established':False,'performance_admitted':False,
    'structural_assumptions':{'target_nvfp4_projections':128,'target_fp8_projections':208,'target_small_bf16_projections':96,
        'mtp_small_bf16_projections':6,'mtp_larger_bf16_projections':2,'linear_layers':48,'full_attention_layers':16},
    'scope':'Partial coverage audit of eleven observed kernel families against generation counters. Initial count audit failed because an extra split-K kernel was counted as another FP8 projection; preserved separately. Every matching primary/follow-up pair now checked in time order on the same graph/stream/device/process/correlation. Prefill counters and its one MTP head are explicitly excluded because markers start after prefill. Structural projection/layer counts and observed kernel-name selectors are source-derived assumptions, not an audit of every CUPTI operation or proof of stable timing/optimization/quality/2x performance.'}
destination.write_text(json.dumps(result,indent=2),encoding='utf8');print(json.dumps(result,indent=2))
