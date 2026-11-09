from audit_half_transport1 import require
def decode_check(events,profile):
    require(profile['fused_decode_rows4'] is True,'missing native decode policy')
    ready=[e for e in events if e['event']=='ready']
    require(len(ready)==1 and 'native_fused_q4_decode_rows4_default_v1' in ready[0]['backend'],'wrong native decode backend')
    prep=[e for e in events if e['event']=='native_fused_q4_decode_prepared']
    require(len(prep)==1 and prep[0]['sites']==[1]*32 and type(prep[0]['profiles']) is int and 1<=prep[0]['profiles']<=2 and prep[0]['scratch_bytes']==65536 and prep[0]['extra_weight_bytes']==0 and events.index(prep[0])<events.index(ready[0]),'wrong native decode preparation')
    pending=None;frames=full=partial=blocks=copies=0
    for event in events:
        if event['event']=='native_batch':
            require(pending is None,'missing native decode observation')
            if event['stage']=='prefill' and not any(q['prefill_tokens'] for q in event['requests']):continue
            pending=event
        elif event['event']=='native_fused_q4_decode':
            require(pending is not None and event['stage']==pending['stage'],'unbound native decode observation')
            actual=len(pending['requests']) if pending['stage']=='decode' else 0
            wanted=32 if actual==4 else 0
            require(type(event['blocks']) is int and event['blocks']==wanted and type(event['input_copies']) is int and 0<=event['input_copies']<=wanted,'wrong native decode route/copies')
            frames+=1;blocks+=wanted;copies+=event['input_copies']
            full+=actual==4;partial+=pending['stage']=='decode' and actual<4;pending=None
    require(pending is None and full>0 and partial>0 and blocks==32*full,'missing full/partial native decode coverage')
    return dict(frames=frames,actual_four_request_decode_frames=full,partial_decode_frames=partial,native_fused_blocks=blocks,native_input_copies=copies,prepared_profiles=prep[0]['profiles'],private_scratch_bytes=65536,extra_weight_bytes=0)

