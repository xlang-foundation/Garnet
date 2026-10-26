"""CPU oracle and lossless full-state block codec for the isolated T128 commit.

No model/runtime imports. GPU outputs are reconstructed exactly, including guards;
the codec records consecutive repeated32-word blocks, never a hash-only witness.
"""
import struct

MAGIC=b'GNP128R1'
GUARD=0x5a65a65a
INITIAL=0x3f003f00
GUARD_WORDS=32
TIES=(0x3f808000,0x3f818000,0xbf808000,0xbf818000,0x00008000,0x80008000,0,0x80000000)

def round_bf16(bits):
    if type(bits) is not int or not 0<=bits<=0xffffffff:
        raise ValueError('A uint32 source is required')
    return ((bits+0x7fff+((bits>>16)&1))&0xffffffff)>>16

def fixture_word(index):
    group=index>>8;dimension=index&31
    if dimension<len(TIES):return TIES[dimension]
    mixed=((group*1664525+dimension*1013904223+17)&0xffffffff)
    mixed^=mixed>>13;mixed=(mixed*0x85ebca6b)&0xffffffff;mixed^=mixed>>16
    return 0x3f000000|(mixed&0x807fffff)

def decode(raw,expected_words):
    import numpy as np
    if len(raw)<16 or raw[:8]!=MAGIC or struct.unpack_from('<Q',raw,8)[0]!=expected_words:
        raise ValueError('Wrong full-state codec header')
    payload=raw[16:]
    if not payload or len(payload)%132 or expected_words%32:
        raise ValueError('Truncated block stream')
    records=np.frombuffer(payload,dtype='<u4').reshape(-1,33)
    counts=records[:,0].astype(np.uint64)
    if not np.all(counts>0) or int(counts.sum())!=expected_words//32:
        raise ValueError('Zero/overflow/missing block run')
    return np.repeat(records[:,1:],counts.astype(np.int64),axis=0).reshape(-1)

def fixture_words(count):
    import numpy as np
    indices=np.arange(count,dtype=np.uint32);group=indices>>8;dimension=indices&31
    with np.errstate(over='ignore'):
        mixed=group*np.uint32(1664525)+dimension*np.uint32(1013904223)+np.uint32(17)
        mixed^=mixed>>13;mixed*=np.uint32(0x85ebca6b);mixed^=mixed>>16
    result=np.uint32(0x3f000000)|(mixed&np.uint32(0x807fffff))
    for d,value in enumerate(TIES):result[dimension==d]=value
    return result

def expected_state(layout,start,count,target):
    import numpy as np
    if not 0<=start<=4096 or not 0<=count<=128 or start+count>4096:
        raise ValueError('Valid T128 span required')
    result=np.full(layout['canonical_state_width']+64,INITIAL,dtype=np.uint32)
    result[:32]=result[-32:]=GUARD
    if not count:return result
    body=result[32:-32];halves=body.view(np.uint16)
    for row in layout['layers']:
        offset=row['offset'];source=row['target_offset']
        if row['kind']=='linear_attention':
            width=row['state1_width']+row['state2_width']
            body[offset:offset+width]=target[source:source+width]
        else:
            for which in range(2):
                words=target[source+which*131072:source+(which+1)*131072].reshape(128,4,256)[:count]
                rounded=((words.astype(np.uint64)+0x7fff+((words>>16)&1))>>16).astype(np.uint16)
                destination=halves[2*(offset+which*row['state1_width']):2*(offset+(which+1)*row['state1_width'])].reshape(4,4096,256)
                destination[:,start:start+count,:]=rounded.transpose(1,0,2)
    return result
