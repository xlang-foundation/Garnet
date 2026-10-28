"""No-model GPU copy/replay correctness. No profiling or speed/memory claim."""
import json,os,struct,sys,time
from pathlib import Path
sys.path.insert(0,os.environ['GARNET_RUNTIME_ROOT'])
import garnet
request_path,output=map(Path,sys.argv[1:3]);assert not output.exists()
request=json.loads(request_path.read_text(encoding='utf-8'));assert sys.byteorder=='little'
io=garnet.cuda_tensor_io();records=[];events=[];error=None
def event(name):
    events.append({'name':name,'pid':os.getpid(),'unix_seconds':time.time()})
    print('PHASE_GPU_EVENT',name,flush=True)
def data(path):return Path(path).read_bytes()
def upload(path,shape):
    assert Path(path).stat().st_size==4*sum_shape(shape)
    return io.read_f32(str(path),shape)
def sum_shape(shape):
    result=1
    for d in shape:result*=d
    return result
def view(root,shape):return io.storage_view(root,shape)
def observed(t):
    words=io.tensor_bytes(t)//4
    if words<=1048576:return io.finite_f32_bytes(t)
    flat=view(t,[1,words]);chunks=[]
    for start in range(0,words,1048576):
        chunk=io.slice(flat,start,[1,min(1048576,words-start)])
        chunks.append(io.finite_f32_bytes(chunk))
    return b''.join(chunks)
def require_error(name,action,state,expected):
    rejected=False;message=None
    try:action()
    except Exception as failure:rejected=True;message=str(failure)
    assert rejected,name
    assert observed(state)==expected,name+' wrote canonical storage'
    return {'name':name,'rejected':True,'canonical_unchanged':True,'error':message}
def changed(root,index,value):
    raw=observed(root);values=list(struct.unpack('<'+str(len(raw)//4)+'f',raw));values[index]=value;io.update(root,values);return raw
def restore(root,raw):
    io.update(root,list(struct.unpack('<'+str(len(raw)//4)+'f',raw)))
event('started')
try:
    for case in request['prefill_cases']:
        b=case['batch'];w=case['state_width'];state_root=upload(case['initial'],[b*w+64]);state=view(state_root,[b,w])
        slots_root=upload(case['slots'],[b*case['tokens']+64]);slots=view(slots_root,[b,case['tokens']])
        active_root=upload(case['active'],[b*case['tokens']+64]);active=view(active_root,[b,case['tokens']])
        checks=[]
        for part in case['parts']:
            spec=part['spec'];p=spec['packed_width'];source_root=upload(part['source'],[b*p+64]);source=view(source_root,[b,p])
            helper=garnet.cuda_packed_prefix()
            try:
                assert helper.phase_output_layout()=='qwen35-phase-outputs-v1'
                helper.build(request['parameters'],json.dumps(spec));stats=json.loads(helper.stats_json())
                assert stats['ready'] and stats['copy_only'] and stats['workspace_bytes']==stats['parameter_bytes']==0
                before=observed(state_root)
                if case['runtime_guards'] and part['part_index']==0:
                    overlap=view(state_root,[b,p])
                    checks.append(require_error('source_canonical_overlap',lambda:helper.commit_all(overlap,state,slots,active),state_root,before))
                    old=changed(source_root,spec['status_offset'],1.0)
                    checks.append(require_error('bad_compact_status',lambda:helper.commit_all(source,state,slots,active),state_root,before));restore(source_root,old)
                    old=changed(active_root,0,2.0)
                    checks.append(require_error('nonbinary_active',lambda:helper.commit_all(source,state,slots,active),state_root,before));restore(active_root,old)
                    old=changed(slots_root,0,0.5)
                    checks.append(require_error('fractional_slot',lambda:helper.commit_all(source,state,slots,active),state_root,before));restore(slots_root,old)
                    checks.append(require_error('rebuild_while_ready',lambda:helper.build(request['parameters'],json.dumps(spec)),state_root,before))
                helper.commit_all(source,state,slots,active)
                # Close immediately, before observing completion, to exercise
                # the existing SDK's leases for asynchronous helper constants.
                helper.close();assert not json.loads(helper.stats_json())['ready'] and json.loads(helper.stats_json())['owned_bytes']==0
                expected=data(part['expected']);assert observed(state_root)==expected,case['name']+' partition state/guard mismatch'
                assert observed(source_root)==data(part['source']) and observed(slots_root)==data(case['slots']) and observed(active_root)==data(case['active'])
                if case['runtime_guards'] and part['part_index']==0:
                    checks.append(require_error('commit_after_close',lambda:helper.commit_all(source,state,slots,active),state_root,expected))
            finally:helper.close()
        records.append({'name':case['name'],'kind':'partition_copy','batch':b,'tokens':case['tokens'],'parts':len(case['parts']),
            'complete_canonical_and_tail_guards_exact':True,'source_slot_active_roots_immutable':True,'helper_immediate_close_owned_zero':True,'runtime_guards':checks,'words':b*w+64})
        print('PHASE_GPU_CASE_PASS',case['name'],flush=True)
    for case in request['replay_cases']:
        b=case['batch'];w=case['state_width'];oldspec=case['legacy_spec'];spec=case['phase_spec']
        initial=upload(case['initial'],[b*w+64])
        modern_root=upload(case['phase_source'],[b*spec['packed_width']+64]);modern=view(modern_root,[b,spec['packed_width']])
        legacy_root=upload(case['legacy_source'],[b*oldspec['packed_width']+64]);legacy=view(legacy_root,[b,oldspec['packed_width']])
        left_root=io.clone(initial);right_root=io.clone(initial);left=view(left_root,[b,w]);right=view(right_root,[b,w])
        h=garnet.cuda_packed_prefix();old=garnet.cuda_packed_prefix()
        try:
            h.build(case['parameters'],json.dumps(spec));old.build(case['parameters'],json.dumps(oldspec))
            assert json.loads(h.stats_json())['ready'] and not json.loads(h.stats_json())['copy_only']
            assert observed(left_root)==observed(right_root)==data(case['initial'])
            for pattern in case['patterns']:
                io.copy_into(initial,left_root);io.copy_into(initial,right_root)
                before=observed(left_root);checks=[]
                if pattern['runtime_guards']:
                    bad=list(pattern['spans']);bad[1]=case['tokens']+1
                    checks.append(require_error('oversized_replay_prefix',lambda:h.commit(modern,left,bad),left_root,before))
                    overlap=view(left_root,[b,spec['packed_width']])
                    checks.append(require_error('replay_source_overlap',lambda:h.commit(overlap,left,pattern['spans']),left_root,before))
                    saved=changed(modern_root,spec['status_offset'],1.0)
                    checks.append(require_error('replay_invalid_status',lambda:h.commit(modern,left,pattern['spans']),left_root,before));restore(modern_root,saved)
                h.commit(modern,left,pattern['spans']);old.commit(legacy,right,pattern['spans'])
                if pattern['last']:
                    h.close();old.close()
                current=observed(left_root);reference=observed(right_root)
                assert current==reference,case['name']+' '+pattern['name']+' complete legacy/phase mismatch'
                assert current[-256:]==data(case['initial'])[-256:]
                if not any(pattern['spans'][1::2]):assert current==before
                for row,count in enumerate(pattern['spans'][1::2]):
                    if count==0:assert current[row*w*4:(row+1)*w*4]==before[row*w*4:(row+1)*w*4]
                assert observed(modern_root)==data(case['phase_source']) and observed(legacy_root)==data(case['legacy_source'])
                records.append({'name':case['name']+'-'+pattern['name'],'kind':'replay_equivalence','batch':b,'spans':pattern['spans'],
                    'all_canonical_words_and_guards_equal_legacy':True,'zero_prefix_rows_unchanged':True,'source_roots_immutable':True,'runtime_guards':checks,'words':b*w+64})
                print('PHASE_GPU_CASE_PASS',records[-1]['name'],flush=True)
            assert json.loads(h.stats_json())['owned_bytes']==json.loads(old.stats_json())['owned_bytes']==0
        finally:h.close();old.close()
    assert len(records)==request['expected_case_count']
except Exception as failure:error=str(failure);raise
finally:
    io.close();event('closed')
    with output.open('x',encoding='utf-8') as f:json.dump({'all_passed':error is None and len(records)==request['expected_case_count'],'error':error,'cases':records,'events':events,
        'case_count':len(records),'model_loaded':False,'speed_or_memory_measured':False,
        'scope':'Synthetic GPU transport and old/new replay-kernel equivalence only. No independent GDN math, compiled graph, rollback/poison transaction or full model qualification.'},f,indent=2)
print('PHASE_GPU_TRANSPORT_COMPLETE',len(records))
