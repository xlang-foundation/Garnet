"""Independent complete dense FP64 outputs for the separate T128 attention study."""
import numpy as np
if __package__:
    from .bf16_kv_overlay_oracle import raw_bits,current_bits,fp32,bf16,spans,attention,ATOL,RTOL
else:
    from bf16_kv_overlay_oracle import raw_bits,current_bits,fp32,bf16,spans,attention,ATOL,RTOL

def case_plan():
    cases=[{'batch':1,'chunk':128,'layer':layer,'end':end} for layer in (3,63) for end in (0,1,127,128,511,512,513,3968,4095,4096)]
    cases += [{'batch':1,'chunk':128,'layer':layer,'end':513} for layer in range(7,63,4)]
    result=[dict(c,id=i) for i,c in enumerate(cases)]
    return result+[dict(batch=1,chunk=128,layer=3,end=128,id=127),dict(batch=1,chunk=128,layer=63,end=4096,id=255)]

def reference_full(case):
    if not isinstance(case,dict) or set(case)!= {'batch','chunk','layer','end','id'} or any(type(v) is not int for v in case.values()) or case not in case_plan():
        raise ValueError('Exact declared T128 attention case required')
    result=np.zeros((1,128,24,256),dtype=np.float64);dims=np.arange(256,dtype=np.uint32)
    start,count=spans(case)[0];end=case['end'];layer=case['layer']
    if not count:return result
    for kh in range(4):
        indices=((kh*4096+np.arange(end,dtype=np.uint32)[:,None])*256+dims)
        key=bf16(raw_bits(indices,1009+layer*131,0x3f000000));value=bf16(raw_bits(indices,2017+layer*131,0x3f000000))
        current=((np.arange(count,dtype=np.uint32)[:,None]*4+kh)*256+dims)
        key[start:start+count]=bf16(current_bits(current,False));value[start:start+count]=bf16(current_bits(current,True))
        rows=[(t,h) for t in range(count) for h in range(6*kh,6*kh+6)]
        qindices=np.stack([(t*24+h)*256+dims for t,h in rows])
        query=fp32(raw_bits(qindices,3119,0x3e800000));gate=fp32(raw_bits(qindices,4127,0x3f800000))
        wanted=attention(query,key,value,gate,np.array([start+t+1 for t,h in rows],dtype=np.int64))
        for (t,h),row in zip(rows,wanted):result[0,t,h]=row
    return result

def compare_full(case,raw):
    actual=np.frombuffer(raw,dtype='<f4').astype(np.float64)
    if actual.size!=128*24*256 or not np.isfinite(actual).all():raise ValueError('Complete finite T128 output required')
    wanted=reference_full(case).reshape(-1);error=np.abs(actual-wanted);fraction=error/(ATOL+RTOL*np.abs(wanted))
    if np.any(fraction>1):raise ValueError('Full FP64 T128 attention mismatch: '+str(case)+', scalar='+str(int(np.argmax(fraction))))
    count=spans(case)[0][1]
    if np.any(actual.reshape(128,24,256)[count:]!=0):raise ValueError('Inactive query output must be exactly zero')
    return {'full_output_scalars':int(actual.size),'active_output_scalars':count*24*256,'max_absolute_error':float(error.max()),
            'max_tolerance_fraction':float(fraction.max()),'inactive_output_exact_zero':True}
