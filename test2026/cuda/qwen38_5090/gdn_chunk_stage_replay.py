"""Independent CPU full-array replay, no GPU/native runtime or speed claims."""
from pathlib import Path
import hashlib
import numpy as np
from .gdn_chunk_stage_layout import layout,pieces,READER_WORDS,STAGES,check_graph
sha=lambda raw:hashlib.sha256(raw).hexdigest()

def array(entry,read):
    assert set(entry)=={'words','sha256','parts'} and type(entry['words']) is int and entry['words']>0
    expected=pieces(entry['words']);assert len(entry['parts'])==len(expected);raw=[]
    for part,(offset,words) in zip(entry['parts'],expected):
        assert set(part)=={'file','offset','words','sha256'} and part['offset']==offset and part['words']==words and type(part['offset']) is type(part['words']) is int
        name=part['file'];assert type(name) is str and Path(name).name==name and name.endswith('.f32')
        value=read(name);assert len(value)==words*4 and sha(value)==part['sha256'];raw.append(value)
    raw=b''.join(raw);assert len(raw)==entry['words']*4 and sha(raw)==entry['sha256'] and np.isfinite(np.frombuffer(raw,dtype='<f4')).all()
    return raw

def replay(report,read):
    assert report['status']=='gdn_chunk_stage_collected_needs_replay' and report['model_or_performance_qualified'] is False
    assert report['prompt_tokens']==3968 and len(report['input_ids'])==128 and len(set(report['input_ids']))>1 and report['input_source']=='first128_actual_text_prompt_embeddings'
    rejected=report['frontend_rejections'];assert len(rejected)==10 and all(row['rejected'] is True and row['error'] for row in rejected)
    assert [row['name'] for row in rejected]==['flag2','flag_negative','flag_bool','flag_float','legacy_conflict','batch2','tokens16','layer2','missing_active','bf16_hidden']
    input_raw=array(report['prompt_input'],read);assert len(input_raw)==128*5120*4
    assert input_raw!=input_raw[:16*5120*4]*8
    assert len(report['cases'])==8
    cases={};words=0
    for row in report['cases']:
        key=(row['layer'],row['tokens'],row['diagnostic']);assert key not in cases and type(row['layer']) is int and row['layer'] in (0,1);cases[key]=row
        spec=layout(row['tokens'],row['diagnostic']);check_graph(row['graph'],row['tokens'],row['diagnostic'])
        assert row['layout']==spec and len(row['repeats'])==2 and row['input_lineage']==('prompt_embeddings' if row['layer']==0 else 'retained_T32_layer0_ordinary_output_common_input')
        active=array(row['active'],read);assert len(active)==row['tokens']*4 and np.array_equal(np.frombuffer(active,dtype='<f4'),np.ones(row['tokens'],dtype='<f4'))
        frames=[]
        for repeat,sequence in enumerate(row['repeats']):
            assert len(sequence)==128//row['tokens'];chunks=[];prior_history=bytes(40960*4);prior_state=bytes(786432*4)
            for index,chunk in enumerate(sequence):
                assert chunk['start']==index*row['tokens'] and chunk['repeat']==repeat and set(chunk['arrays'])=={part['name'] for part in spec['rows']}
                hidden=array(chunk['input_hidden'],read);history=array(chunk['input_history'],read);state=array(chunk['input_state'],read)
                assert array(chunk['input_history_after'],read)==history and array(chunk['input_state_after'],read)==state and array(chunk['input_hidden_after'],read)==hidden
                source=input_raw if row['layer']==0 else reference_layer0
                assert hidden==source[index*row['tokens']*5120*4:(index+1)*row['tokens']*5120*4] and history==prior_history and state==prior_state
                values={name:array(entry,read) for name,entry in chunk['arrays'].items()};words+=sum(len(value)//4 for value in values.values())
                primary=values['primary'];assert len(primary)==spec['primary_words']*4
                h=row['tokens']*5120*4;prior_history=primary[h:h+40960*4];prior_state=primary[h+40960*4:]
                if row['diagnostic']:
                    assert values['masked_input']==hidden and values['output_hidden']==primary[:h]
                    assert all(len(values[name])==row['tokens']*width*4 for name,width in STAGES)
                chunks.append(values)
            if repeat:assert chunks==frames[0]
            frames.append(chunks)
        row['_frames']=frames[0]
        if key==(0,32,False):reference_layer0=b''.join(chunk['primary'][:32*5120*4] for chunk in frames[0])
    # Mandatory collection order establishes common layer1 input before it is used.
    assert [(row['layer'],row['tokens'],row['diagnostic']) for row in report['cases']]==[(layer,t,diag) for layer in (0,1) for t in (32,128) for diag in (False,True)]
    comparisons=[]
    for layer in (0,1):
        for t in (32,128):
            normal=cases[(layer,t,False)]['_frames'];diagnostic=cases[(layer,t,True)]['_frames'];assert len(normal)==len(diagnostic)
            assert all(a['primary']==b['primary'] for a,b in zip(normal,diagnostic)),('Diagnostic primary changes ordinary',layer,t)
        a=cases[(layer,32,True)]['_frames'];b=cases[(layer,128,True)]['_frames'][0];different=[]
        for name,width in STAGES:
            left=b''.join(frame[name] for frame in a);right=b[name];assert len(left)==len(right)==128*width*4
            x=np.frombuffer(left,dtype='<u4');y=np.frombuffer(right,dtype='<u4');changed=np.flatnonzero(x!=y)
            if changed.size:different.append({'stage':name,'different_f32_words':int(changed.size),'first_different_word':int(changed[0]),'total_words':int(x.size)})
        comparisons.append({'layer':layer,'input_lineage':'prompt_embeddings' if layer==0 else 'common_retained_T32_layer0_output','different_stages':different,'first_different_stage':different[0]['stage'] if different else None,'final_history_exact':a[-1]['primary'][32*5120*4:32*5120*4+40960*4]==b['primary'][128*5120*4:128*5120*4+40960*4],'final_state_exact':a[-1]['primary'][32*5120*4+40960*4:]==b['primary'][128*5120*4+40960*4:]})
    for row in report['cases']:del row['_frames']
    return {'status':'gdn_chunk_stage_primary_witness_and_arrays_replayed','profiles':8,'frontend_rejections':10,'repeats_per_profile':2,'complete_output_f32_words_replayed':words,'diagnostic_primary_exact_to_ordinary':True,'all_input_and_carried_state_words_exact':True,'stage_comparisons':comparisons,'full3968_model_cause_established':False,'model_or_quality_or_performance_qualified':False,'two_times_target_achieved':False}
