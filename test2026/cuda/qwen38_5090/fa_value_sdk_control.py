"""Actual FA3/11 value-cache baselines before any SDK intervention is admitted."""
import hashlib,math,struct
import numpy as np
from .decode_boundary_analysis import validate_observations
from .prefill128_fp8_projection import input_oracle
from .fa_value_sdk_log import parse_log
CASES=(('v3',3,5),('v11',11,1));K=5120;N=1024;PARENT='decode-boundaries-attempt1'
sha=lambda raw:hashlib.sha256(raw).hexdigest()

def verify_derivations(record,read):
    for row in record['derivations']:
        raw=read(row['source']);assert sha(raw)==row['source_sha256']
        for change in row['changes']:assert raw.count(change['old'].encode())==change['count'];raw=raw.replace(change['old'].encode(),change['new'].encode())
        if 'retained_function' in row:
            assert row['retained_function']=='parse_log' and sha(raw)==row['intermediate_sha256'];start=raw.index(b'def parse_log(raw):');assert sha(raw[start:])==row['retained_function_sha256'];raw=row['final_prefix'].encode()+raw[start:]
        assert raw==read(row['target']) and sha(raw)==row['target_sha256']

def parent_fixture(reference,candidate,read_reference,read_candidate):
    r=validate_observations(reference,read_reference);c=validate_observations(candidate,read_candidate);data={};cases=[]
    for label,layer,step in CASES:
        inputs=[];small=[];large=[]
        for row in range(4):
            a,b=r[step,row],c[step,row];assert reference['rows'][row]['generated_ids'][0][:step]==candidate['rows'][0]['generated_ids'][row][:step]
            assert a['boundaries'][layer-1]==b['boundaries'][layer-1] and a['normalizations'][layer]==b['normalizations'][layer]
            inputs.append(a['normalizations'][layer]);small.append(a['value_cache'][layer]);large.append(b['value_cache'][layer])
        raw=b''.join(inputs);expected4=b''.join(small);expected16=b''.join(large);assert len(raw)==16*K*4 and raw!=raw[:4*K*4]*4 and len(expected4)==len(expected16)==4*N*2
        changed=int(np.count_nonzero(np.frombuffer(expected4,dtype='<u2')!=np.frombuffer(expected16,dtype='<u2')));assert changed==1
        stem=label+'-step0';data[stem+'-normalized.f32']=raw;data[stem+'-garnet-m4-active.bf16']=expected4;data[stem+'-garnet-m16-active.bf16']=expected16
        cases.append(dict(label=label,layer=layer,actual_decode_step=step,physical_rows=[0,1,2,3],same_consumed_inputs_and_full_preceding_hidden_and_separate_norm=True,actual_active_V_BF16_different_words=changed,input_sha256=sha(raw),reference_active_sha256=sha(expected4),candidate_active_sha256=sha(expected16)))
    return data,cases

def active_values(raw):
    assert len(raw)==16*N*2
    return np.frombuffer(raw,dtype='<u2').reshape(4,4,N)[:,0,:].copy().tobytes()

def observe(read_fixture,read_actual):
    rows=[]
    for label,layer,step in CASES:
        scale_raw=read_fixture(label+'-input-scale.f32');assert len(scale_raw)==4;scale=struct.unpack('<f',scale_raw)[0];assert math.isfinite(scale) and scale>0
        stem=label+'-step0';source=read_fixture(stem+'-normalized.f32');assert len(source)==16*K*4;cast,packed=input_oracle(source,scale)
        assert read_actual(stem+'-cast.bf16')==cast and read_actual(stem+'-packed4.e4m3')==read_actual(stem+'-packed16.e4m3')==packed
        expected4=read_fixture(stem+'-garnet-m4-active.bf16');expected16=read_fixture(stem+'-garnet-m16-active.bf16');assert len(expected4)==len(expected16)==4*N*2;prior={}
        for repeat in range(2):
            values={mode:read_actual(stem+'-'+mode+'-r'+str(repeat)+'.bf16') for mode in ('m4','default16','shared16')}
            for mode,raw in values.items():
                assert len(raw)==16*N*2 and np.all((np.frombuffer(raw,dtype='<u2')&0x7f80)!=0x7f80)
                if repeat:assert raw==prior[mode],'SDK repeat instability'
            prior=values
            def count(a,b):return int(np.count_nonzero(np.frombuffer(a,dtype='<u2')!=np.frombuffer(b,dtype='<u2')))
            rows.append(dict(projection=label,layer=layer,actual_decode_step=step,repeat=repeat,full_output_words=16*N,active_output_words=4*N,Garnet_M4_active_values_reproduced=active_values(values['m4'])==expected4,Garnet_M16_active_values_reproduced=active_values(values['default16'])==expected16,default_M16_vs_four_M4_full_different_words=count(values['default16'],values['m4']),default_M16_vs_four_M4_active_different_words=count(active_values(values['default16']),active_values(values['m4'])),M4_algorithm_at_M16_full_different_words=count(values['shared16'],values['m4']),M4_algorithm_at_M16_active_different_words=count(active_values(values['shared16']),active_values(values['m4']))))
    return dict(status='actual_FA_value_SDK_full_word_observation',cases=rows,CPU_input_cast_and_FP8_quantization_exact=True,four_M4_and_default_M16_reproduce_actual_Garnet_active_cache=all(v['Garnet_M4_active_values_reproduced'] and v['Garnet_M16_active_values_reproduced'] for v in rows),same_M4_algorithm_at_M16_full_words_exact=all(v['M4_algorithm_at_M16_full_different_words']==0 for v in rows),full_Garnet_inactive_V_projection_outputs_captured=False,internal_fused_normalization_directly_captured=False,full_model_or_quality_or_performance_qualified=False)

def replay(read_fixture,read_actual):
    result=observe(read_fixture,read_actual)
    assert result['four_M4_and_default_M16_reproduce_actual_Garnet_active_cache'],'SDK baseline did not reproduce both actual Garnet active V caches'
    assert result['same_M4_algorithm_at_M16_full_words_exact'],'M4-selected algorithm at actual M16 changed full SDK output words'
    return {**result,'status':'actual_FA_value_SDK_baselines_and_M4_algorithm_at_M16_exact'}
