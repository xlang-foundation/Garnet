"""Complete finite same-input projection observations; no performance or model promotion."""
import hashlib,math
import numpy as np
from .gdn_chunk_stage_replay import array
STAGES=(('normalized',5120),('key',1024),('value',1024))
sha=lambda b:hashlib.sha256(b).hexdigest()
def check_graph(stats,tokens):
    assert type(tokens) is int and tokens in (32,128) and stats['checkpoint_weight_precision']=='modelopt_mixed'
    for key,value in [('native_fp8_projection_weights',2),('native_nvfp4_projection_weights',0),('native_int4_g128_projection_weights',0),('external_compressed_weight_inputs',0),('external_bf16_weight_inputs',0),('native_fp8_m128_canonical_projections',0),('native_fp8_z128_canonical_projections',0),('native_fp8_z128_reconstructed_projections',0),('device_output_bytes',tokens*7168*4),('graph_capture_failures',0),('graph_captures',1)]:
        assert type(stats[key]) is int and stats[key]==value,(key,stats[key])
    assert type(stats['graph_replays']) is int and stats['graph_replays']>=1
def weight_metadata(rows):
    assert set(rows)=={'model.language_model.layers.3.self_attn.k_proj.weight','model.language_model.layers.3.self_attn.v_proj.weight'}
    for row in rows.values():
        for k,v in [('logical_shape',[1024,5120]),('packed_shape',[1024,1280]),('packed_dtype','INT32'),('packed_bytes',5242880),('recipe','modelopt_fp8_w8a8'),('scale_bytes',0),('scale_shape',[]),('joined_mlp',False)]:assert type(row[k]) is type(v) and row[k]==v,(k,row[k])
        for k in ('weight_scale','input_scale'):assert type(row[k]) in (int,float) and math.isfinite(row[k]) and row[k]>0
def difference(a,b,width):
    assert len(a)==len(b)==128*width*4
    x=np.frombuffer(a,dtype='<u4');y=np.frombuffer(b,dtype='<u4');changed=np.flatnonzero(x!=y)
    af=np.frombuffer(a,dtype='<f4').astype(np.float64);bf=np.frombuffer(b,dtype='<f4').astype(np.float64);delta=bf-af;norm=np.linalg.norm(af)
    return {'byte_exact':a==b,'complete_F32_words_compared':128*width,'changed_F32_words':int(changed.size),'first_difference':dict(token=int(changed[0])//width,dimension=int(changed[0])%width,T32_bits=int(x[changed[0]]),T128_bits=int(y[changed[0]])) if changed.size else None,'max_absolute_difference':float(np.max(np.abs(delta))),'relative_l2_difference':float(np.linalg.norm(delta)/norm) if norm else (0.0 if not changed.size else None)}
def replay(report,read,fixture):
    assert report['status']=='actual_FA3_projection_witness_collected' and report['model_or_performance_qualified'] is False and report['witness_policy']=='1'
    assert report['source_input_metadata']['sha256']==report['input_fixture_sha256']==sha(fixture) and len(fixture)==2621440 and array(report['common_input'],read)==fixture
    weight_metadata(report['weight_metadata']);assert len(report['cases'])==2 and [r['tokens'] for r in report['cases']]==[32,128]
    frames={};entries=1;words=0
    for row in report['cases']:
        t=row['tokens'];check_graph(row['graph'],t);assert len(row['repeats'])==2;prior=None
        for repeat,sequence in enumerate(row['repeats']):
            assert len(sequence)==128//t;outputs={name:[] for name,width in STAGES}
            for index,chunk in enumerate(sequence):
                assert chunk['start']==index*t and chunk['repeat']==repeat and set(chunk['arrays'])=={n for n,w in STAGES}
                expected=fixture[index*t*5120*4:(index+1)*t*5120*4];assert array(chunk['input_hidden'],read)==array(chunk['input_hidden_after'],read)==expected;entries+=2
                for name,width in STAGES:
                    raw=array(chunk['arrays'][name],read);assert len(raw)==t*width*4;outputs[name].append(raw);entries+=1;words+=len(raw)//4
            actual={n:b''.join(v) for n,v in outputs.items()}
            if prior is not None:assert actual==prior,'Repeated projection outputs changed'
            prior=actual
        frames[t]=prior
    assert entries==51 and words==3670016
    stages={name:difference(frames[32][name],frames[128][name],width) for name,width in STAGES}
    return {'status':'actual_FA3_projection_complete_arrays_replayed','finite_array_entries':entries,'complete_output_F32_words_replayed':words,'same_real_input':True,'readonly_inputs_exact':True,'repeats_exact':True,'stages':stages,'normalized_exact':stages['normalized']['byte_exact'],'key_value_both_exact':all(stages[n]['byte_exact'] for n in ('key','value')),'actual_projection_difference_with_identical_normalized_input':stages['normalized']['byte_exact'] and not all(stages[n]['byte_exact'] for n in ('key','value')),'full_attention_cache_cause_established':False,'model_or_quality_or_performance_qualified':False,'two_times_target_achieved':False}
