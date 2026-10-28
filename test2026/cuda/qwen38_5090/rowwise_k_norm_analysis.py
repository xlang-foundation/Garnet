"""Ordinary-output controls for source87; no performance or full-context admission."""
import json
from .b4_k35_cache52 import primary_parity,checked_cache
from .b4_k35_coordinate_trajectory import checked_trajectory,compare
from .b4_k_m4x4_control52 import intervention
PHASES=('off-reference','off-candidate','on-candidate')

def norm_trace(raw,enabled):
    if isinstance(raw,bytes):raw=raw.decode('utf8',errors='replace')
    rows=[json.loads(line.split(' ',1)[1]) for line in raw.splitlines() if line.startswith('GARNET_ROWWISE_K_DECODE_NORM ')]
    assert not any(line.startswith('GARNET_DECODE35_STAGE_WITNESS ') or line.startswith('GARNET_FULL_CHUNK_STAGE_WITNESS ') for line in raw.splitlines())
    if enabled:
        assert len(rows)==16 and {r['layer'] for r in rows}==set(range(3,64,4))
        assert all(r==dict(batch=4,tokens=4,layer=r['layer'],rms_subgraphs=4) for r in rows)
    else:assert not rows
    return dict(enabled=enabled,full_attention_layers=len(rows),rms_subgraphs_per_layer=4 if enabled else 0,executed_kernel_count_claimed=False)

def control(report,parent,read,read_parent,new_library,old_library):
    assert report['library_sha256']==new_library and parent['library_sha256']==old_library
    assert report['intervention_policy']=='off' and report['primary_outputs_only'] is True
    batch=report['physical_batch'];assert batch in (1,4)
    # The DLL transition is explicit; all numerical fields remain strict.
    result=primary_parity(report,dict(parent,library_sha256=new_library),read,read_parent)
    row,old=report['rows'][0],parent['rows'][0];position=0 if batch==1 else 2
    a=checked_cache(row['key35_cache52'],row['final_state']['logical']['components'][70],batch,position,read)
    b=checked_cache(old['key35_cache52'],old['final_state']['logical']['components'][70],batch,position,read_parent)
    assert a==b
    x=checked_trajectory(row['key35_coordinate_trajectory'],a,batch,position)
    y=checked_trajectory(old['key35_coordinate_trajectory'],b,batch,position);assert x==y
    return dict(status='source87_defaultoff_all_primary_cache_trajectory_exact',primary=result,new_library_sha256=new_library,historical_library_sha256=old_library,raw_cache_bytes=len(a),coordinate_timepoints=53,numerical_tolerance_applied=False,model_or_performance_qualified=False)

def analyze(selection,key,reports,reads,full_reference,prefix_candidate,read_full,read_prefix,library):
    assert set(reports)==set(reads)==set(PHASES)
    ref,off,on=(reports[n] for n in PHASES)
    assert all(r['library_sha256']==library and r['primary_outputs_only'] is True for r in reports.values())
    assert ref['intervention_policy']==off['intervention_policy']=='off' and on['intervention_policy']=='on'
    assert on['physical_batch']==off['physical_batch']==4 and ref['physical_batch']==1
    for field in ('status','mode','capacity','prefill_chunk','decode_chunk','inputs','graph_stats','vision_stats'):
        assert on[field]==off[field],('On/off profile drift',field)
    keys=selection['groups'][2]['keys'];assert keys[2]==key and [r['keys'] for r in on['rows']]==[keys]
    gold={r['keys'][0]:r['generated_ids'][0][:53] for r in full_reference['rows']}
    actual=on['rows'][0]['generated_ids'];assert len(actual)==4
    for a,k in zip(actual,keys):assert len(a)==len(gold[k])==53 and all(type(v) is int and 0<=v<248320 for v in a)
    first=[next((i for i,(a,b) in enumerate(zip(values,gold[k])) if a!=b),None) for values,k in zip(actual,keys)]
    result=dict(status='ordinary_rowwise_K_ablation_observed',all_four_first_differing_ID_indices=first,all_four_consumed_52_IDs_exact=all(i is None or i>=52 for i in first),selected_step52_intervention_passed=False,full128_output_4K_MTP_quality_HTTP_or_performance_qualified=False,original_full4k_still_failed=True,two_times_target_achieved=False)
    # A changed consumed trajectory invalidates any same-input cache inference.
    if not result['all_four_consumed_52_IDs_exact']:
        result['interpretation']='Consumed tokens diverged; no same-input cache or unique-operator inference.'
        return result
    result['intervention']=intervention(selection,key,ref,on,full_reference,prefix_candidate,reads['off-reference'],reads['on-candidate'],read_full,read_prefix)
    result['selected_step52_intervention_passed']=result['intervention']['selected_step52_intervention_passed']
    points=[]
    for phase,batch,position in (('off-reference',1,0),('off-candidate',4,2),('on-candidate',4,2)):
        row=reports[phase]['rows'][0];raw=checked_cache(row['key35_cache52'],row['final_state']['logical']['components'][70],batch,position,reads[phase]);points.append(checked_trajectory(row['key35_coordinate_trajectory'],raw,batch,position))
    result['off_coordinate_comparison']=compare(points[0],points[1]);result['on_coordinate_comparison']=compare(points[0],points[2])
    return result
