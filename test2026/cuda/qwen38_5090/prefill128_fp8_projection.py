"""Independent CPU input conversion and complete BF16 word replay; no speed admission."""
import math,struct
import numpy as np

def input_oracle(raw,scale):
    assert len(raw)%4==0 and np.isfinite(np.frombuffer(raw,dtype='<f4')).all() and math.isfinite(float(scale)) and scale>0
    words=np.frombuffer(raw,dtype='<u4');cast=((words+np.uint32(0x7fff)+((words>>16)&1))>>16).astype('<u2')
    bf32=(cast.astype('<u4')<<16).view('<f4');assert np.isfinite(bf32).all()
    scaled=bf32/np.float32(scale);codes=np.arange(127,dtype=np.uint8);exponent=(codes.astype(np.int32)>>3)&15;mantissa=codes&7
    table=np.where(exponent==0,np.ldexp(mantissa.astype(np.float32),-9),np.ldexp(1+mantissa.astype(np.float32)/8,exponent-7)).astype(np.float32)
    values=np.minimum(np.abs(scaled),np.float32(448));upper=np.minimum(np.searchsorted(table,values),126);lower=np.maximum(upper-1,0)
    dl,du=values-table[lower],table[upper]-values
    chosen=np.where((du<dl)|((du==dl)&((upper&1)==0)),upper,lower).astype(np.uint8);chosen|=np.signbit(scaled).astype(np.uint8)<<7
    return cast.tobytes(),chosen.tobytes()

def replay_projection(read_fixture,read_actual,mode):
    assert mode in ('control','shared');source=read_fixture('input-m16-k6144.f32');base=read_fixture('expected-m16-n5120.bf16')
    assert len(source)==16*6144*4 and len(base)==16*5120*2
    assert read_fixture('input-m32-k6144.f32')==source*2
    scale_raw=read_fixture('input-scale.f32');assert len(scale_raw)==4;scale=struct.unpack('<f',scale_raw)[0]
    result=[]
    for m in (16,32,128):
        raw=source*(m//16);cast,quantized=input_oracle(raw,scale)
        assert read_actual('cast-m'+str(m)+'.bf16')==cast and read_actual('quantized-m'+str(m)+'.e4m3')==quantized
        repeated=base*(m//16);expected=read_fixture('expected-m'+str(m)+'-n5120.bf16') if mode=='control' and m in (16,32) else repeated
        assert len(expected)==m*5120*2;prior=None
        for repeat in (0,1):
            actual=read_actual('actual-m'+str(m)+'-r'+str(repeat)+'.bf16');assert len(actual)==m*5120*2
            words=np.frombuffer(actual,dtype='<u2');assert np.all((words&0x7f80)!=0x7f80)
            if repeat:assert actual==prior
            prior=actual
            if mode=='shared' or m in (16,32):assert actual==expected
            result.append({'m':m,'repeat':repeat,'output_words':m*5120,'different_from_repeated_m16_output_words':int(np.count_nonzero(words!=np.frombuffer(repeated,dtype='<u2'))),'archived_output_exact':actual==expected if m in (16,32) and mode=='control' else None,'m128_default_is_observation_only':mode=='control' and m==128,'shared_algorithm_output_exact':actual==repeated if mode=='shared' else None})
    return {'mode':mode,'cases':result,'complete_bf16_output_words_replayed':sum(row['output_words'] for row in result),'input_cast_and_quantization_independently_exact':True,'model_or_performance_qualified':False}
