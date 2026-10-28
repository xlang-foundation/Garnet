"""Independent NumPy FP64 attention reference; no CUDA/Garnet imports.

Production uses warp/block FP32 reductions. This reference constructs complete
visible key/value matrices and uses dense double precision matrix arithmetic.
BF16 values are constructed by integer round-to-nearest-even, including ties.
"""
import numpy as np

HEADS=(0,5,6,12,23)
ATOL=3e-5
RTOL=2e-5

def case_plan():
    result=[{'batch':b,'chunk':t,'layer':layer,'end':end}
            for b in (4,8) for t in (4,32) for layer in (3,63)
            for end in (0,1,511,512,513,4095,4096)]
    result += [{'batch':8,'chunk':4,'layer':layer,'end':513} for layer in range(7,63,4)]
    return [dict(case,id=i) for i,case in enumerate(result)]

def raw_bits(indices,salt,exponent):
    u=np.asarray(indices,dtype=np.uint32)*np.uint32(1664525)+np.uint32(salt)
    u=u^(u>>13);u=u*np.uint32(0x85ebca6b);u=u^(u>>16)
    return np.uint32(exponent)|(u&np.uint32(0x807fffff))

def current_bits(indices,value):
    i=np.asarray(indices,dtype=np.uint32);d=i%256
    ties=np.array([0x3f808000,0x3f818000,0xbf808000,0xbf818000],dtype=np.uint32)
    return np.where(d<4,ties[np.minimum(d,3)],raw_bits(i,6197 if value else 5171,0x3f000000)).astype(np.uint32)

def fp32(raw):
    return np.asarray(raw,dtype=np.uint32).view(np.float32).astype(np.float64)

def bf16(raw):
    u=np.asarray(raw,dtype=np.uint32)
    rounded=((u+np.uint32(0x7fff)+((u>>16)&1))>>16)<<16
    return fp32(rounded)

def spans(case):
    b,t,end=case['batch'],case['chunk'],case['end']
    counts=[min(t,end,1+(row*7+case['id'])%t) if end else 0 for row in range(b)]
    return [(end-count,count) for count in counts]

def sample_triples(case):
    return [(b,t,h) for b,(_,count) in enumerate(spans(case))
            if count for t in sorted({0,count//2,count-1}) for h in HEADS]

def attention(query,key,value,gate,visible):
    query,key,value,gate=map(lambda a:np.asarray(a,dtype=np.float64),(query,key,value,gate))
    visible=np.asarray(visible)
    if query.ndim!=2 or key.ndim!=2 or value.shape!=key.shape or gate.shape!=query.shape or query.shape[1]!=key.shape[1]:
        raise ValueError('Oracle matrix shapes disagree')
    if visible.shape!=(len(query),) or not np.issubdtype(visible.dtype,np.integer) or np.any(visible<1) or np.any(visible>len(key)):
        raise ValueError('Oracle visible prefixes invalid')
    if not all(np.isfinite(a).all() for a in (query,key,value,gate)):
        raise ValueError('Oracle inputs must be finite')
    scores=(query@key.T)/np.sqrt(query.shape[1])
    scores[np.arange(len(key))[None,:]>=visible[:,None]]=-np.inf
    weights=np.exp(scores-np.max(scores,axis=1,keepdims=True))
    weights/=weights.sum(axis=1,keepdims=True)
    return (weights@value)/(1+np.exp(-gate))

def reference_case(case):
    end,chunk,layer=case['end'],case['chunk'],case['layer'];result={}
    dims=np.arange(256,dtype=np.uint32)
    for b,(start,count) in enumerate(spans(case)):
        if not count:continue
        ts=sorted({0,count//2,count-1})
        for kh in range(4):
            heads=[h for h in HEADS if h//6==kh]
            indices=(((b*4+kh)*4096+np.arange(end,dtype=np.uint32)[:,None])*256+dims)
            key=bf16(raw_bits(indices,1009+layer*131,0x3f000000))
            value=bf16(raw_bits(indices,2017+layer*131,0x3f000000))
            current=(((b*chunk+np.arange(count,dtype=np.uint32)[:,None])*4+kh)*256+dims)
            key[start:start+count]=bf16(current_bits(current,False))
            value[start:start+count]=bf16(current_bits(current,True))
            rows=[(t,h) for t in ts for h in heads]
            qindices=np.stack([((b*chunk+t)*24+h)*256+dims for t,h in rows])
            query=fp32(raw_bits(qindices,3119,0x3e800000))
            gate=fp32(raw_bits(qindices,4127,0x3f800000))
            expected=attention(query,key,value,gate,np.array([start+t+1 for t,h in rows],dtype=np.int64))
            result.update({(b,t,h):row for (t,h),row in zip(rows,expected)})
    return result

def compare_witness(case,raw):
    triples=sample_triples(case)
    actual=np.frombuffer(raw,dtype='<f4').astype(np.float64)
    if actual.size!=len(triples)*256 or not np.isfinite(actual).all():
        raise ValueError('Malformed/nonfinite numerical witnesses')
    if not triples:return {'samples':0,'scalars':0,'max_absolute_error':0.0,'max_tolerance_fraction':0.0}
    expected=reference_case(case);wanted=np.stack([expected[triple] for triple in triples]).reshape(-1)
    error=np.abs(actual-wanted);fraction=error/(ATOL+RTOL*np.abs(wanted))
    if np.any(fraction>1):
        i=int(np.argmax(fraction));raise ValueError(f'FP64 oracle mismatch: case={case}, scalar={i}, actual={actual[i]}, expected={wanted[i]}, fraction={fraction[i]}')
    return {'samples':len(triples),'scalars':actual.size,'max_absolute_error':float(error.max()),'max_tolerance_fraction':float(fraction.max())}
