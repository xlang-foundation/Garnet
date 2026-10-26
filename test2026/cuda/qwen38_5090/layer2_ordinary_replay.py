"""Independent complete finite-array replay for actual controlled layer2 ordinary outputs."""
import hashlib
import numpy as np
from .gdn_chunk_stage_replay import array
from .layer2_ordinary_layout import INPUT_SOURCE,validate_profile,check_graph
sha=lambda raw:hashlib.sha256(raw).hexdigest()

def extract_common_input(report,read):
    assert report['status']=='gdn_chunk_stage_collected_needs_replay' and report['model_or_performance_qualified'] is False
    rows=[row for row in report['cases'] if (row['layer'],row['tokens'],row['diagnostic'])==(1,32,False)];assert len(rows)==1
    row=rows[0];assert len(row['repeats'])==2 and row['input_lineage']=='retained_T32_layer0_ordinary_output_common_input'
    values=[]
    for repeat,chunks in enumerate(row['repeats']):
        assert len(chunks)==4;current=[]
        for index,chunk in enumerate(chunks):
            assert chunk['start']==index*32 and chunk['repeat']==repeat
            primary=array(chunk['arrays']['primary'],read);assert len(primary)==(32*5120+40960+786432)*4;current.append(primary[:32*5120*4])
        values.append(b''.join(current))
    assert values[0]==values[1] and len(values[0])==128*5120*4 and values[0]!=values[0][:16*5120*4]*8
    return values[0]

def replay(report,read,fixture):
    assert report['status']=='layer2_ordinary_collected_needs_replay' and report['model_or_performance_qualified'] is False
    assert report['prompt_tokens']==3968 and len(report['input_ids'])==128 and len(set(report['input_ids']))>1 and report['input_source']==INPUT_SOURCE
    assert report['frontend_rejections']==[] and len(report['cases'])==2
    source=array(report['common_input'],read);assert source==fixture and len(source)==128*5120*4 and sha(source)==report['input_fixture_sha256']
    frames={};entries=1;words=0
    for row in report['cases']:
        tokens=row['tokens'];spec=validate_profile(row['layer'],tokens,row['diagnostic']);assert row['layout']==spec and row['input_lineage']==INPUT_SOURCE and tokens not in frames
        check_graph(row['graph'],tokens,False);active=array(row['active'],read);assert np.array_equal(np.frombuffer(active,dtype='<f4'),np.ones(tokens,dtype='<f4'));entries+=1
        assert len(row['repeats'])==2;repeats=[]
        for repeat,sequence in enumerate(row['repeats']):
            assert len(sequence)==128//tokens;chunks=[];history=bytes(40960*4);state=bytes(786432*4)
            for i,chunk in enumerate(sequence):
                assert chunk['start']==i*tokens and chunk['repeat']==repeat and set(chunk['arrays'])=={'primary'}
                hidden=array(chunk['input_hidden'],read);oldh=array(chunk['input_history'],read);olds=array(chunk['input_state'],read)
                assert hidden==source[i*tokens*5120*4:(i+1)*tokens*5120*4] and oldh==history and olds==state
                for name,value in (('input_hidden_after',hidden),('input_history_after',oldh),('input_state_after',olds)):assert array(chunk[name],read)==value
                primary=array(chunk['arrays']['primary'],read);assert len(primary)==spec['primary_words']*4;words+=len(primary)//4;entries+=7
                h=tokens*5120*4;history=primary[h:h+40960*4];state=primary[h+40960*4:];chunks.append(primary)
            if repeats:assert chunks==repeats[0]
            repeats.append(chunks)
        frames[tokens]=repeats[0]
    assert [row['tokens'] for row in report['cases']]==[32,128] and entries==73 and words==10895360
    left=b''.join(raw[:32*5120*4] for raw in frames[32]);right=frames[128][0][:128*5120*4];x=np.frombuffer(left,dtype='<u4');y=np.frombuffer(right,dtype='<u4');changed=np.flatnonzero(x!=y)
    a=np.frombuffer(left,dtype='<f4').astype(np.float64);b=np.frombuffer(right,dtype='<f4').astype(np.float64);delta=b-a;norm=np.linalg.norm(a)
    h32=32*5120*4;h128=128*5120*4
    return {'status':'layer2_ordinary_complete_arrays_replayed','profiles':2,'finite_array_entries':entries,'complete_output_f32_words_replayed':words,'same_real_common_input':True,'readonly_inputs_and_carried_states_exact':True,'repeats_exact':True,'hidden_words':655360,'hidden_bytes_exact':left==right,'different_hidden_f32_words':int(changed.size),'first_hidden_difference':{'token':int(changed[0])//5120,'dimension':int(changed[0])%5120,'T32_bits':int(x[changed[0]]),'T128_bits':int(y[changed[0]])} if changed.size else None,'hidden_max_absolute_difference':float(np.max(np.abs(delta))),'hidden_relative_l2_difference':float(np.linalg.norm(delta)/norm) if norm else (0.0 if not changed.size else None),'final_history_exact':frames[32][-1][h32:h32+40960*4]==frames[128][0][h128:h128+40960*4],'final_state_exact':frames[32][-1][h32+40960*4:]==frames[128][0][h128+40960*4:],'full_model_cause_established':False,'model_or_quality_or_performance_qualified':False,'two_times_target_achieved':False}
