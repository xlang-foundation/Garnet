"""Full connected-component cache baselines; original SDK/model failures stay failed."""
import hashlib,math,re,struct
import numpy as np
from .fa_value_sdk_control import CASES,N,K
from .decode_boundary_analysis import blob
from .decode_fp8_sdk_control import bf16_output
sha=lambda raw:hashlib.sha256(raw).hexdigest()
COUNTS=('native_fp8_m128_canonical_projections','native_fp8_z128_canonical_projections','native_fp8_z128_reconstructed_projections','native_fp8_kv128_canonical_projections','native_fp8_q128_canonical_projections','native_fp8_o128_canonical_projections','native_full_chunk_stage_witnesses')
def observe(reference,candidate,read_reference,read_candidate,read_fixture):
    reports={'reference':reference,'candidate':candidate};collected={};rows=[]
    for mode,report in reports.items():
        batch=1 if mode=='reference' else 4;read=read_reference if batch==1 else read_candidate
        assert report['status']=='connected_FA_value_components_collected' and report['mode']==mode and type(report['physical_batch']) is int and report['physical_batch']==batch and report['tokens']==4 and report['capacity']==4096
        assert report['model_or_performance_qualified'] is False and report['internal_fused_normalization_directly_captured'] is False
        assert report['cache_mode']=='ordinary_zero_FP32_inputs_component_only' and report['position_and_slot_controls']=='zero_component_controls_V_projection_independent'
        assert len(report['cases'])==(8 if batch==1 else 2)
        for record in report['cases']:
            label=record['label'];matching=[v for v in CASES if v[0]==label];assert len(matching)==1;_,layer,step=matching[0];assert record['layer']==layer and record['actual_decode_step']==step
            physical=record['physical_rows'];assert len(physical)==batch and all(type(v) is int and v in range(4) for v in physical)
            assert physical==([physical[0]] if batch==1 else [0,1,2,3])
            stats=record['graph_stats'];assert type(stats['native_fp8_projection_weights']) is type(stats['native_nvfp4_projection_weights']) is type(stats['external_compressed_weight_inputs']) is int and stats['native_fp8_projection_weights']==4 and stats['native_nvfp4_projection_weights']==2 and stats['external_compressed_weight_inputs']==6
            assert all(type(stats[key]) is type(record['norm_stats'][key]) is int and stats[key]==0 and record['norm_stats'][key]==0 for key in COUNTS)
            assert record['norm_stats']['native_fp8_projection_weights']==record['norm_stats']['native_nvfp4_projection_weights']==0
            assert record['input_cache_sha256_before']==record['input_cache_sha256_after'] and len(record['input_cache_sha256_before'])==2*batch
            zero_hash=sha(bytes(4194304*4));assert record['input_cache_sha256_before']==[zero_hash]*(2*batch)
            expected_input=read_fixture(label+'-step0-hidden.f32');expected_norm=read_fixture(label+'-step0-normalized.f32')
            selected_input=b''.join(expected_input[r*81920:(r+1)*81920] for r in physical);selected_norm=b''.join(expected_norm[r*81920:(r+1)*81920] for r in physical)
            actual_input=blob(record['input'],read,batch*81920);assert actual_input==selected_input
            assert record['input_after_sha256']==record['input']['sha256'] and len(record['repeats'])==2
            prior=None
            for repeat,entry in enumerate(record['repeats']):
                assert entry['repeat']==repeat and entry['input_after_sha256']==record['input']['sha256']
                normalized=blob(entry['unmasked_normalized'],read,batch*81920);hidden=blob(entry['component_hidden'],read,batch*81920);value=blob(entry['component_V'],read,batch*1024*4)
                assert normalized==selected_norm,'Unchanged standalone normalizer did not reproduce parent full arrays'
                packed=bf16_output(value,batch*1024);current=(normalized,hidden,value)
                if prior is not None:assert current==prior,'Connected component repeat instability'
                prior=current
            for position,row in enumerate(physical):
                key=(label,row);assert key not in collected;collected[key]=packed[position*2048:(position+1)*2048]
        assert set(collected)=={(label,row) for label,_,_ in CASES for row in range(4)}
        rows.append(dict(mode=mode,values=dict(collected)));collected={}
    a,b=rows[0]['values'],rows[1]['values'];comparisons=[]
    for label,layer,step in CASES:
        expected={mode:read_fixture(label+'-step0-garnet-'+mode+'-active.bf16') for mode in ('m4','m16')};assert all(len(v)==8192 for v in expected.values())
        for row in range(4):
            x=a[label,row];y=b[label,row];r=expected['m4'][row*2048:(row+1)*2048];c=expected['m16'][row*2048:(row+1)*2048]
            def count(u,v):return int(np.count_nonzero(np.frombuffer(u,dtype='<u2')!=np.frombuffer(v,dtype='<u2')))
            comparisons.append(dict(label=label,layer=layer,actual_decode_step=step,physical_row=row,B1_component_vs_actual_B1_cache_different_words=count(x,r),B4_component_vs_actual_B4_cache_different_words=count(y,c),B4_component_vs_four_B1_different_words=count(y,x)))
    return dict(status='connected_FA_value_components_full_active_cache_observed',rows=comparisons,all_original_cache_baselines_exact=all(r['B1_component_vs_actual_B1_cache_different_words']==r['B4_component_vs_actual_B4_cache_different_words']==0 for r in comparisons),all_standalone_normalizer_full_arrays_exact=True,all_component_hidden_and_V_repeats_exact=True,all_complete_zero_input_KV_caches_readonly=True,internal_fused_normalization_directly_captured=False,full_model_or_quality_or_performance_qualified=False)
def admit(*args):
    result=observe(*args);assert result['all_original_cache_baselines_exact'],'Connected component did not reproduce all actual B1/B4 Vcache values'
    return {**result,'status':'connected_FA_value_component_baselines_exact_only'}
