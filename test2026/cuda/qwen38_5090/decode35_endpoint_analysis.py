"""Six endpoints over the unchanged native seven-stage hook; full ordinary controls precede interpretation."""
import re,struct
from .b4_b1_exact_gate import _logical,_vectors
from .b4_full4k_selection52 import STATUS,ids,observe
from .b4_k35_cache52 import checked_cache
from .b4_k35_coordinate_trajectory import checked_trajectory
from .decode_boundary_analysis import blob,difference

OPERANDS=(('masked_input',20480),('input_normalized',20480),('projected_Q_gate',49152),('projected_K',4096),('projected_V',4096),('K_post_RoPE',4096))
PHASES=('off-reference','off-candidate','on-reference','on-candidate')

def checked_layers(report,read,key,enabled):
    batch=report['physical_batch'];assert type(batch) is int and batch in (1,4)
    assert len(report['rows'])==1 and report['boundary_runs']==52
    p=report['observer'];assert set(p)=={'layers','tokens','hidden','hidden_width','primary_width','wide_width','selected_steps'}
    assert p['layers']==list(range(64)) and p['selected_steps']==[44,45]
    assert (p['tokens'],p['hidden'],p['hidden_width'])==(4,5120,20480)
    assert type(p['primary_width']) is int and 0<p['primary_width']<2147483647-64*20480-102400
    assert p['wide_width']==p['primary_width']+64*20480+(102400 if enabled else 0)
    row=report['rows'][0];position=row['keys'].index(key);assert (batch,position) in ((1,0),(4,2))
    assert len(row['keys'])==len(set(row['keys']))==batch and len(row['generated_ids'])==batch
    assert [e['step'] for e in row['layer_observations']]==[44,45]
    result={}
    for e in row['layer_observations']:
        step=e['step'];assert type(step) is int and e['program_runs']==step
        assert e['consumed_tokens']==[v[step-1] for v in row['generated_ids']]
        assert set(e['boundaries'])=={str(i) for i in range(64)}
        assert len({v['file'] for v in e['boundaries'].values()})==64
        raw={i:blob(e['boundaries'][str(i)],read,batch*81920) for i in range(64)}
        result[step]={i:b[position*81920:(position+1)*81920] for i,b in raw.items()}
    return result

def primary_control(current,parent,read,read_parent,key,current_library,parent_library,current_enabled=False,library_transition=False):
    """Declared DLL transition is permitted only for off controls; numeric gates stay exact."""
    for h in (current_library,parent_library):assert isinstance(h,str) and re.fullmatch('[0-9a-f]{64}',h)
    assert current['library_sha256']==current_library and parent['library_sha256']==parent_library
    assert type(library_transition) is bool and (current_library!=parent_library)==library_transition
    assert not current_enabled or not library_transition
    for name in ('status','mode','physical_batch','capacity','prefill_chunk','decode_chunk','inputs'):assert current[name]==parent[name],('primary metadata drift',name)
    assert current['status']==STATUS and current['operand_witness'] is current_enabled
    batch=current['physical_batch'];assert (batch,current['mode']) in ((1,'reference'),(4,'candidate'))
    assert [r['keys'] for r in current['rows']]==[r['keys'] for r in parent['rows']] and len(current['rows'])==1
    c,p=current['rows'][0],parent['rows'][0];ids(c['generated_ids'],p['generated_ids'],batch,53)
    assert c['prompt_cache']==p['prompt_cache']
    assert [v['step'] for v in c['post_step_states']]==[v['step'] for v in p['post_step_states']]==[7,52]
    assert c['final_state']==c['post_step_states'][-1]['capture'] and p['final_state']==p['post_step_states'][-1]['capture']
    pairs=[(c[k],p[k]) for k in ('reset_state','prefill_state','final_state','final_reset')]+[(c['post_step_states'][0]['capture'],p['post_step_states'][0]['capture'])]
    for a,b in pairs:
        _logical(a['logical'],batch);_logical(b['logical'],batch)
        assert a['logical']==b['logical'],'Observer or DLL changed complete primary state'
        assert _vectors(a['vectors'],read,batch)==_vectors(b['vectors'],read_parent,batch),'Observer or DLL changed full primary vector'
    for k in ('embedding_mode','head_precision','compressed_projection_pool_entries','compressed_nvfp4_entries','compressed_fp8_entries'):assert current['pool'][k]==parent['pool'][k]
    position=0 if batch==1 else 2
    raw=checked_cache(c['key35_cache52'],c['final_state']['logical']['components'][70],batch,position,read)
    oldraw=checked_cache(p['key35_cache52'],p['final_state']['logical']['components'][70],batch,position,read_parent)
    assert raw==oldraw,'Observer or DLL changed full selected cache'
    checked_trajectory(c['key35_coordinate_trajectory'],raw,batch,position)
    checked_trajectory(p['key35_coordinate_trajectory'],oldraw,batch,position)
    assert c['key35_coordinate_trajectory']==p['key35_coordinate_trajectory'],'Observer or DLL changed coordinate timeline'
    a=checked_layers(current,read,key,current_enabled);b=checked_layers(parent,read_parent,key,False)
    assert a==b,'Observer or DLL changed recorded layer output'
    return {'status':'53IDs_five_complete_primary_captures_full_cache_timeline_all64_layers_exact','physical_batch':batch,'current_library_sha256':current_library,'parent_library_sha256':parent_library,'declared_off_DLL_transition':library_transition,'complete_captures':5,'components_per_row':128,'layers':64,'layer_steps':[44,45],'tolerance_applied':False}

def witness_trace(raw,batch,enabled):
    assert type(batch) is int and batch in (1,4) and type(enabled) is bool
    import json
    rows=[]
    for line in raw.decode('utf8',errors='replace').splitlines():
        if line.startswith('GARNET_DECODE35_STAGE_WITNESS '):rows.append(json.loads(line.split(' ',1)[1]))
    expected={'batch':batch,'tokens':4,'layer':35,'stages':7,'row_words':106496}
    assert rows==([expected] if enabled else []),'Unexpected native decode35 pack trace'
    return {'status':'one_actual_decode35_pack_trace' if enabled else 'decode35_pack_trace_absent','events':len(rows),'profile':expected if enabled else None}

def bf16_rn(raw):
    assert isinstance(raw,bytes) and len(raw)%4==0
    result=bytearray()
    for (u,) in struct.iter_unpack('<I',raw):
        assert u&0x7f800000!=0x7f800000,'Nonfinite F32 K operand'
        v=((u+0x7fff+((u>>16)&1))>>16)&0xffff
        assert v&0x7f80!=0x7f80,'Nonfinite rounded BF16 K operand'
        result.extend(struct.pack('<H',v))
    return bytes(result)

def checked_operands(report,read,key):
    assert report['operand_witness'] is True
    batch=report['physical_batch'];position=0 if batch==1 else 2
    assert report['rows'][0]['keys'].index(key)==position
    m=report['internal_observer'];assert set(m)=={'names','widths','offsets','row_words'}
    assert m['names']==[n for n,_ in OPERANDS] and m['widths']==[w for _,w in OPERANDS] and m['row_words']==102400
    cursor=report['observer']['primary_width']+64*20480
    expected=[]
    for name,width in OPERANDS:expected.append([name,cursor,width]);cursor+=width
    assert m['offsets']==expected and report['observer']['wide_width']==cursor
    row=report['rows'][0];entries=row['operand_observations'];assert [e['step'] for e in entries]==[44,45]
    cache=checked_cache(row['key35_cache52'],row['final_state']['logical']['components'][70],batch,position,read)
    boundary=checked_layers(report,read,key,True);result={}
    for e in entries:
        step=e['step'];assert type(step) is int and e['program_runs']==step
        assert e['consumed_tokens']==[v[step-1] for v in row['generated_ids']]
        assert set(e['operands'])=={n for n,_ in OPERANDS} and len({v['file'] for v in e['operands'].values()})==6
        values={n:blob(e['operands'][n],read,batch*w*4)[position*w*4:(position+1)*w*4] for n,w in OPERANDS}
        x=values['masked_input'];assert x[:20480]==boundary[step][34][:20480] and x[20480:]==bytes(61440),'Captured masked input does not link to exact active layer34 output'
        post=values['K_post_RoPE']
        slot=3968+step-1;expected_cache=b''.join(cache[(head*4096+slot)*512:(head*4096+slot+1)*512] for head in range(4))
        assert bf16_rn(post[:4096])==expected_cache,'Captured active K does not round to actual full cache slot'
        result[step]=values
    return result

def analyze(selection,key,reports,old_reports,reads,old_reads,current_library,old_libraries):
    assert set(reports)==set(reads)==set(PHASES) and set(old_reports)==set(old_reads)==set(old_libraries)=={'reference','candidate'}
    controls={}
    # All numeric controls across both rows/profiles must pass before any operand read.
    for role in ('reference','candidate'):
        off,on='off-'+role,'on-'+role
        controls[off]=primary_control(reports[off],old_reports[role],reads[off],old_reads[role],key,current_library,old_libraries[role],False,True)
        controls[on]=primary_control(reports[on],reports[off],reads[on],reads[off],key,current_library,current_library,True,False)
    observations={}
    for prefix in ('off','on'):
        observations[prefix]=observe(selection,key,reports[prefix+'-reference'],reports[prefix+'-candidate'],lambda k,n:reads[prefix+'-reference'](n),reads[prefix+'-candidate'])
    r=checked_operands(reports['on-reference'],reads['on-reference'],key);c=checked_operands(reports['on-candidate'],reads['on-candidate'],key)
    rows=[];key_path=['masked_input','input_normalized','projected_K','K_post_RoPE']
    for step in (44,45):
        changes={n:difference(r[step][n],c[step][n],active_words=w//4) for n,w in OPERANDS}
        samples={}
        for n in ('projected_K','K_post_RoPE'):
            samples[n]={'head':0,'dimension':211,'reference_F32':struct.unpack_from('<f',r[step][n],211*4)[0],'candidate_F32':struct.unpack_from('<f',c[step][n],211*4)[0]}
        rows.append({'step':step,'operands':changes,'first_differing_active_operand':next((n for n,_ in OPERANDS if changes[n]['active_different_words']),None),'first_differing_active_key_path_operand':next((n for n in key_path if changes[n]['active_different_words']),None),'coordinate_samples':samples})
    return {'status':'decode35_endpoints_compared_after_reused_off_historical_and_same_DLL_on_parity','controls':controls,'ordinary_cross_profile_observations':observations,'rows':rows,'actual_K_cache_BF16_rounding_link_passed':True,'unique_causal_operator_or_first_global_drift_proved':False,'original_full4K_still_failed':True,'model_quality_HTTP_or_performance_qualified':False,'two_times_target_achieved':False}
