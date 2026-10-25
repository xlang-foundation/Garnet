"""Independent fixture replay using vectorized values and first-index maxima."""
from functools import lru_cache
import numpy as np

SHAPES=((1,1),(1,7),(4,255),(4,256),(4,1023),(4,1024),(4,1025),
        (4,8191),(4,8192),(1,248320),(4,248320),(8,248320),
        (16,248320),(128,248320),(4,248321))


@lru_cache(maxsize=None,typed=True)
def row_oracle(row,vocab,pattern,dtype):
    if any(type(n) is not int for n in (row,vocab,pattern)) or not 0<=row<128 or not 1<=vocab<=248321 or not 0<=pattern<11 or dtype not in ('fp32','bf16'):
        raise ValueError('Unqualified fixed top1 fixture')
    if pattern in (1,2,3,5,6,7,8,9) and row:
        return row_oracle(0,vocab,pattern,dtype)
    if pattern==4 and row>1:return row_oracle(row%2,vocab,pattern,dtype)
    if pattern==0:
        col=np.arange(vocab,dtype=np.uint32)
        seed=(col*np.uint32(1664525)+np.uint32(1013904223))^np.uint32((row*2246822519)&0xffffffff)
        data=((seed%np.uint32(2047)).astype(np.int32)-1023).astype(np.float32)/np.float32(64)
        raw=data.view(np.uint32)
    elif pattern==4:
        raw=((np.arange(vocab,dtype=np.uint32)+np.uint32(row))%np.uint32(2))*np.uint32(0x80000000)
    else:
        fill={1:0x7fc00000,2:0xff800000,3:0xff7fffff,5:0xbf800000,
              6:0x7fc00000,7:0xc0400000,8:0x80000001,9:0x7fc00000,10:0xc4fa0000}[pattern]
        raw=np.full(vocab,fill,dtype=np.uint32)
        if pattern==5:raw[[min(vocab-1,1024),vocab-1]]=0x7f800000
        if pattern==6:raw[-1]=0xbf800000
        if pattern==7:raw[[min(vocab-1,1023),min(vocab-1,1024),vocab-1]]=0x45800000
        if pattern==8:raw[-1]=1
        if pattern==9:raw[-1]=0xff7fffff
        if pattern==10:raw[[(row*257)%vocab,vocab-1]]=0xc47a0000
    if dtype=='bf16':
        # Integer round-to-nearest-even, independent of CUDA conversion and
        # its host intrinsic. These fixtures never select a NaN payload.
        widened=raw.astype(np.uint64)
        raw=((widened+0x7fff+((widened>>16)&1))&0xffff0000).astype(np.uint32)
    values=raw.view(np.float32);initial=np.float32(-np.finfo(np.float32).max)
    eligible=values>initial
    if not np.any(eligible):return 0,0xff7fffff
    maximum=np.max(values[eligible])
    index=int(np.flatnonzero(values==maximum)[0])
    return index,int(raw[index])


def expected_rows(dtype,rows,vocab,pattern):
    if type(rows) is not int or (rows,vocab) not in SHAPES:raise ValueError('Unknown fixed top1 shape')
    values=[row_oracle(row,vocab,pattern,dtype) for row in range(rows)]
    return [v[0] for v in values],[v[1] for v in values]
