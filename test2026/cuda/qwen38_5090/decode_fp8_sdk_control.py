"""Full-array CPU replay for the actual M4/M16 SDK intervention."""
import math,re,struct
import numpy as np
from .prefill128_fp8_projection import input_oracle
PROJECTIONS={'qkv':10240,'z':6144}
K=5120

def bf16_output(raw,words):
    assert len(raw)==words*4
    bits=np.frombuffer(raw,dtype='<u4');assert np.isfinite(bits.view('<f4')).all() and np.all((bits&65535)==0)
    return (bits>>16).astype('<u2').tobytes()

def observe(read_fixture,read_actual):
    rows=[]
    for projection,n in PROJECTIONS.items():
        scale=struct.unpack('<f',read_fixture(projection+'-input-scale.f32'))[0];assert math.isfinite(scale) and scale>0
        for step in range(5):
            stem=projection+'-step'+str(step);source=read_fixture('step'+str(step)+'-normalized.f32');assert len(source)==16*K*4
            cast,packed=input_oracle(source,scale);assert read_actual(stem+'-cast.bf16')==cast and read_actual(stem+'-packed16.e4m3')==read_actual(stem+'-packed4.e4m3')==packed
            expected4=read_fixture(stem+'-garnet-m4.bf16');expected16=read_fixture(stem+'-garnet-m16.bf16');assert len(expected4)==len(expected16)==16*n*2
            prior={}
            for repeat in range(2):
                values={mode:read_actual(stem+'-'+mode+'-r'+str(repeat)+'.bf16') for mode in ('m4','default16','shared16')}
                for mode,raw in values.items():
                    assert len(raw)==16*n*2 and np.all((np.frombuffer(raw,dtype='<u2')&0x7f80)!=0x7f80)
                    if repeat:assert raw==prior[mode],'SDK repeat instability'
                prior=values
                changed=int(np.count_nonzero(np.frombuffer(values['default16'],dtype='<u2')!=np.frombuffer(values['m4'],dtype='<u2')))
                shared_changed=int(np.count_nonzero(np.frombuffer(values['shared16'],dtype='<u2')!=np.frombuffer(values['m4'],dtype='<u2')))
                rows.append(dict(projection=projection,step=step,repeat=repeat,words=16*n,default_M16_vs_four_M4_different_words=changed,M4_algorithm_at_M16_different_words=shared_changed,Garnet_M4_reproduced=values['m4']==expected4,Garnet_M16_reproduced=values['default16']==expected16))
    return dict(status='actual_M4_M16_SDK_full_word_observation',cases=rows,CPU_input_cast_and_FP8_quantization_exact=True,four_M4_and_default_M16_reproduce_Garnet=all(v['Garnet_M4_reproduced'] and v['Garnet_M16_reproduced'] for v in rows),same_M4_algorithm_at_M16_exact=all(v['M4_algorithm_at_M16_different_words']==0 for v in rows),full_model_or_quality_or_performance_qualified=False)

def replay(read_fixture,read_actual):
    result=observe(read_fixture,read_actual)
    assert result['four_M4_and_default_M16_reproduce_Garnet'],'SDK baseline did not reproduce original Garnet M4/M16'
    assert result['same_M4_algorithm_at_M16_exact'],'M4-selected algorithm at actual M16 did not preserve all reference words'
    return {**result,'status':'actual_M4_M16_SDK_same_algorithm_full_word_control_replayed'}

def parse_log(raw):
    if isinstance(raw,bytes):raw=raw.decode('utf8')
    assert 'CUBLASLT_VERSION 130400' in raw and 'SM120' in raw and 'ACTUAL_M4_M16_SDK_CAPTURE_COMPLETE_NO_MODEL_OR_PERFORMANCE_ADMISSION' in raw
    pattern=r'ALGO projection=(qkv|z) step=(\d+) repeat=(\d+) row=(\d+) mode=(m4|default16|shared16) m=(\d+) status=(-?\d+) id=(-?\d+) tile=(-?\d+) stages=(-?\d+) split=(-?\d+) reduction=(-?\d+) swizzle=(-?\d+) custom=(-?\d+) workspace=(\d+)'
    calls=[];configs={};seen=set()
    for match in re.finditer(pattern,raw):
        p,s,r,row,mode,m,status,*rest=match.groups();s,r,row,m,status=map(int,(s,r,row,m,status));values=list(map(int,rest));attrs=values[:7];workspace=values[7]
        assert s in range(5) and r in (0,1) and status==0 and workspace==4<<20 and m==(4 if mode=='m4' else 16) and row in (range(4) if mode=='m4' else range(1))
        key=(p,s,r,row,mode);assert key not in seen;seen.add(key)
        if mode=='m4':
            if p in configs:assert configs[p]==attrs
            else:configs[p]=attrs
        if mode=='shared16':assert configs[p]==attrs
        calls.append(dict(projection=p,step=s,repeat=r,row=row,mode=mode,M=m,attributes=dict(zip(('id','tile','stages','split','reduction','swizzle','custom'),attrs)),workspace=workspace))
    assert len(calls)==120 and set(configs)==set(PROJECTIONS)
    checks=re.findall(r'ALGO_CHECK projection=(qkv|z) step=(\d+) repeat=(\d+) m=(\d+) status=(-?\d+) state=(-?\d+) workspace=(\d+) capacity=(\d+)',raw);assert len(checks)==20
    checkkeys=set()
    for p,s,r,m,status,state,workspace,capacity in checks:
        assert (p,int(s),int(r)) not in checkkeys;checkkeys.add((p,int(s),int(r)));assert int(s) in range(5) and int(r) in (0,1) and int(m)==16 and int(status)==int(state)==0 and int(workspace)<=int(capacity)==4<<20
    return dict(status='actual_M4_M16_SDK_attributes_and_AlgoCheck_replayed',calls=calls,actual_M16_AlgoChecks=20,reference_algorithms=configs,all_seven_attributes_queried=True,no_unsupported_or_workspace_fallback=True)
