"""Independent bounded carrier/cache observations; no runtime imports or timing claims."""
import array,hashlib,math,struct,sys
from pathlib import PurePosixPath
from .bf16_packed_state_gate import same_packed_bf16_row

PREFIXES=(128,256,512,1024,2048,3968)
WORDS=2097152
BLOCK=1048576
FIELDS=('native_fp8_m128_canonical_projections','native_fp8_z128_canonical_projections')
sha=lambda raw:hashlib.sha256(raw).hexdigest()

def cache_components(layout):
    if len(layout)!=64:raise ValueError('Expected complete64layer layout')
    layer=layout[3]
    if (layer['index'],layer['kind'],layer['offset'],layer['state1_width'],layer['state2_width'])!=(3,'full_attention',3*(40960+786432),WORDS,WORDS):
        raise ValueError('Expected pinned layer3 packedBF16 canonical layout')
    return {'key':layer['offset'],'value':layer['offset']+WORDS}

def parts(words=WORDS):
    if type(words) is not int or words<1:raise ValueError('Invalid read width')
    return [(start,min(BLOCK,words-start)) for start in range(0,words,BLOCK)]

def read_cache(entry,read):
    assert set(entry)=={'words','sha256','parts'} and entry['words']==WORDS
    rows=[]
    assert len(entry['parts'])==2
    for row,(offset,words) in zip(entry['parts'],parts()):
        assert set(row)=={'file','offset','words','sha256'} and type(row['offset']) is int and type(row['words']) is int
        assert (row['offset'],row['words'])==(offset,words)
        path=PurePosixPath(row['file']);assert path.name==row['file'] and path.suffix=='.f32'
        raw=read(row['file']);assert len(raw)==words*4 and sha(raw)==row['sha256'];rows.append(raw)
    raw=b''.join(rows);assert sha(raw)==entry['sha256']
    return raw

def same_state(actual,reference,read,oldread):
    assert same_packed_bf16_row(actual['logical'],0,reference['logical'],0)
    assert set(actual['vectors'])==set(reference['vectors'])=={'hidden','normalized'}
    for name in ('hidden','normalized'):
        for row,fetch in ((actual['vectors'][name],read),(reference['vectors'][name],oldread)):
            assert set(row)=={'file','bytes','sha256'} and PurePosixPath(row['file']).name==row['file']
            raw=fetch(row['file']);assert len(raw)==row['bytes']==20480 and sha(raw)==row['sha256']
            assert all(math.isfinite(v) for v in struct.unpack('<5120f',raw))
        assert read(actual['vectors'][name]['file'])==oldread(reference['vectors'][name]['file'])
    return True

def cache_difference(actual,reference,prefix):
    assert type(prefix) is int and prefix in PREFIXES and len(actual)==len(reference)==WORDS*4 and sys.byteorder=='little'
    x=array.array('H');x.frombytes(actual);y=array.array('H');y.frombytes(reference)
    changed=active=inactive=0;first=None;by_head=[0]*4;maximum=0.0
    for i,(v,w) in enumerate(zip(x,y)):
        assert (v&0x7f80)!=0x7f80 and (w&0x7f80)!=0x7f80,'Nonfinite BF16 cache'
        head=i//(4096*256);token=(i//256)%4096;dimension=i%256
        if token>=prefix:assert v==w==0,'Inactive canonical cache tail must remain zero'
        if v!=w:
            changed+=1;by_head[head]+=1
            if token<prefix:active+=1
            else:inactive+=1
            if first is None:first={'head':head,'token':token,'dimension':dimension,'actual_bf16_bits':v,'reference_bf16_bits':w}
            a=struct.unpack('<f',struct.pack('<I',v<<16))[0];b=struct.unpack('<f',struct.pack('<I',w<<16))[0];maximum=max(maximum,abs(a-b))
    return {'exact':actual==reference,'changed_bf16_values':changed,'active_values_changed':active,'inactive_values_changed':inactive,'changed_by_head':by_head,'first_difference':first,'maximum_absolute_difference':maximum,'complete_finite_bf16_values_compared':WORDS*2,'inactive_tail_zero':True}

def check_graphs(graphs,tokens):
    assert type(tokens) is int and tokens in (32,128) and set(graphs)=={'4',str(tokens)}
    for chunk,row in graphs.items():
        for field in FIELDS:
            value=row[field];assert type(value) is int and value==(48 if tokens==128 and chunk=='128' else 0)
    return True
