"""Read-only complete key-cache bytes, tied to the ordinary state witness."""
import hashlib,re,struct
from .bf16_float_reference_state import expand_bf16_carrier
from .b4_b1_exact_gate import _logical,_vectors
from .b4_full4k_selection52 import observe,ids,STATUS

LAYER=35
HEADS,CAPACITY,DIM=4,4096,256
WORDS=2097152
CACHE_BYTES=8388608
sha=lambda raw:hashlib.sha256(raw).hexdigest()

def key_layout(layout):
    assert len(layout)==64;cursor=0
    for i,layer in enumerate(layout):
        full=i%4==3;w1,w2=(WORDS,WORDS) if full else (40960,786432)
        assert (layer['index'],layer['kind'],layer['offset'],layer['state1_width'],layer['state2_width'])==(i,'full_attention' if full else 'linear_attention',cursor,w1,w2)
        cursor+=w1+w2
    assert cursor==106823680 and layout[LAYER]['offset']==55894016
    return dict(layer=LAYER,role=1,offset=layout[LAYER]['offset'],physical_words=WORDS,logical_words=WORDS*2,heads=HEADS,capacity=CAPACITY,dimension=DIM)

def capture_key(io,state,layout,batch,position,out,label,component):
    assert type(batch) is type(position) is int and (batch,position) in ((1,0),(4,2))
    spec=key_layout(layout);assert (component['layer'],component['kind'],component['role'],component['logical_words'])==(35,'full_attention',1,4194304)
    assert len(component['row_sha256'])==batch
    part=io.zeros([batch,1048576]);values=[]
    for repeat in range(2):
        pieces=[]
        for offset in (0,1048576):
            io.slice_into(state,spec['offset']+offset,part);raw=bytes(io.finite_f32_bytes(io.row(part,position)));assert len(raw)==4194304;pieces.append(raw)
        values.append(b''.join(pieces))
        name=label+'-key35.bf16carrier';path=out/name
        if repeat==0:
            assert not path.exists();path.write_bytes(values[0])
        elif values[1]!=values[0]:
            extra=out/(label+'-key35-repeat1.bf16carrier');assert not extra.exists();extra.write_bytes(values[1])
    assert values[0]==values[1] and len(values[0])==CACHE_BYTES
    expanded_sha=sha(expand_bf16_carrier(values[0]));assert expanded_sha==component['row_sha256'][position]
    return dict(file=name,bytes=CACHE_BYTES,sha256=sha(values[0]),repeat_sha256=[sha(v) for v in values],logical_sha256=expanded_sha,physical_row=position,geometry=spec,storage='little_endian_two_BF16_per_F32_carrier',step=52)

def checked_cache(entry,component,batch,position,read):
    assert type(batch) is type(position) is int and (batch,position) in ((1,0),(4,2))
    assert re.fullmatch(r'[A-Za-z0-9_.-]+\.bf16carrier',entry['file']) and entry['bytes']==CACHE_BYTES
    assert entry['step']==52 and entry['physical_row']==position and entry['storage']=='little_endian_two_BF16_per_F32_carrier'
    assert entry['geometry']==dict(layer=35,role=1,offset=55894016,physical_words=WORDS,logical_words=WORDS*2,heads=4,capacity=4096,dimension=256)
    assert (component['layer'],component['kind'],component['role'],component['logical_words'])==(35,'full_attention',1,4194304) and len(component['row_sha256'])==batch
    raw=read(entry['file']);assert isinstance(raw,bytes) and len(raw)==CACHE_BYTES and sha(raw)==entry['sha256']
    assert entry['repeat_sha256']==[entry['sha256']]*2
    assert sha(expand_bf16_carrier(raw))==entry['logical_sha256']==component['row_sha256'][position]
    return raw

def primary_parity(report,parent,read,read_parent):
    for key in ('status','mode','physical_batch','capacity','prefill_chunk','decode_chunk','library_sha256','inputs'):assert report[key]==parent[key]
    assert report['status']==STATUS and [r['keys'] for r in report['rows']]==[r['keys'] for r in parent['rows']]
    batch=report['physical_batch'];captures=0
    for row,old in zip(report['rows'],parent['rows']):
        ids(row['generated_ids'],old['generated_ids'],batch,53)
        assert [v['step'] for v in row['post_step_states']]==[7,52]
        pairs=[(row[phase],old[phase]) for phase in ('reset_state','prefill_state','final_state','final_reset')]
        pairs.append((row['post_step_states'][0]['capture'],old['post_step_states'][0]['capture']))
        for a,b in pairs:
            _logical(a['logical'],batch);_logical(b['logical'],batch);assert a['logical']==b['logical']
            assert _vectors(a['vectors'],read,batch)==_vectors(b['vectors'],read_parent,batch);captures+=1
    return dict(status='all_current_primary_endpoints_step7_step52_vectors_and53_IDs_exact',physical_batch=batch,complete_captures=captures,components_per_row=128,performance_qualified=False)

def compare_cache(reference,candidate):
    import numpy as np
    assert isinstance(reference,bytes) and isinstance(candidate,bytes) and len(reference)==len(candidate)==CACHE_BYTES
    a,b=np.frombuffer(reference,dtype='<u2'),np.frombuffer(candidate,dtype='<u2')
    assert not np.any((a&0x7f80)==0x7f80) and not np.any((b&0x7f80)==0x7f80),'Nonfinite BF16 key value'
    changed=np.flatnonzero(a!=b);heads=changed//(CAPACITY*DIM);slots=(changed//DIM)%CAPACITY;dimensions=changed%DIM
    decode=(slots>=3968)&(slots<4020);prompt=slots<3968;inactive=slots>=4020
    samples=[]
    for i,head,slot,dim in zip(changed[:16],heads[:16],slots[:16],dimensions[:16]):
        x,y=int(a[i]),int(b[i]);samples.append(dict(head=int(head),slot=int(slot),dimension=int(dim),reference_bits=x,candidate_bits=y,reference_value=struct.unpack('<f',struct.pack('<I',x<<16))[0],candidate_value=struct.unpack('<f',struct.pack('<I',y<<16))[0],scheduled_decode_writer_step=(int(slot)-3968+1 if 3968<=slot<4020 else None)))
    return dict(status='all_key35_cache_words_compared',BF16_values=4194304,different_BF16_values=int(changed.size),different_by_head=np.bincount(heads,minlength=4).tolist(),different_by_slot=np.bincount(slots,minlength=4096).tolist(),different_prompt_values=int(np.count_nonzero(prompt)),different_decode_values=int(np.count_nonzero(decode)),different_inactive_values=int(np.count_nonzero(inactive)),first_differing_slot=(int(slots.min()) if changed.size else None),earliest_scheduled_decode_writer_step=(int(slots[decode].min())-3968+1 if np.any(decode) else None),samples=samples,all_bits_exact=not bool(changed.size),numerical_tolerance_applied=False,scope='All4heads/4096slots/256dimensions compared asBF16bits, includingprompt/inactive slots. Scheduledwriterstep derivesfromslot, notproof oftime oruniqueoperator thatchangedthevalue.')

def analyze(selection,key,reference,candidate,old_reference,old_candidate,read_reference,read_candidate,read_old_reference,read_old_candidate):
    parity={mode:primary_parity(report,parent,read,oldread) for mode,report,parent,read,oldread in (('reference',reference,old_reference,lambda n:read_reference(None,n),lambda n:read_old_reference(None,n)),('candidate',candidate,old_candidate,read_candidate,read_old_candidate))}
    original=observe(selection,key,reference,candidate,read_reference,read_candidate)
    r,c=reference['rows'][0],candidate['rows'][0]
    a=checked_cache(r['key35_cache52'],r['final_state']['logical']['components'][70],1,0,lambda n:read_reference(key,n))
    b=checked_cache(c['key35_cache52'],c['final_state']['logical']['components'][70],4,2,read_candidate)
    return dict(status='key35_raw_cache_compared_after_complete_primary_parity',selected_key=key,physical_row=2,primary_parity=parity,ordinary_step52_observation=original,cache_comparison=compare_cache(a,b),original_full4k_admission_still_failed=True,MTP_quality_HTTP_or_performance_qualified=False,scope='Complete raw8MiBkey35carriers twice-read andlinkedbyexpandedSHA256toactualcompletecomponent70witness. Originalmodelgraphs/weights/nativeDLLs unchanged. Fivecompleteprimarycaptures/53IDs matchordinarysameprofileparent forselectedB1/allfourB4rows before interpretation. No uniqueoperatorcausality/numeric/performanceadmission.')
