"""Trace one proved affected row, retaining whole-B4 ordinary primary parity."""
from .all_layer_full4k_decode_analysis import primary_parity,layers,LAYERS
from .b4_full4k_decode_prefix import observe
from .decode_boundary_analysis import difference

def analyze_selected_row(selection,selected_key,reference,candidate,ordinary_reference,ordinary_candidate,read_reference,read_candidate,read_ordinary_reference,read_ordinary_candidate):
    keys=selection['groups'][2]['keys'];assert selected_key in keys
    assert [row['keys'] for row in reference['rows']]==[[selected_key]]
    assert [row['keys'] for row in candidate['rows']]==[keys]
    subset=dict(ordinary_reference,rows=[row for row in ordinary_reference['rows'] if row['keys']==[selected_key]])
    assert len(subset['rows'])==1
    parity={mode:primary_parity(report,parent,read,old_read) for mode,report,parent,read,old_read in (
        ('reference',reference,subset,lambda n:read_reference(None,n),lambda n:read_ordinary_reference(None,n)),
        ('candidate',candidate,ordinary_candidate,read_candidate,read_ordinary_candidate))}
    original=observe(selection,ordinary_reference,ordinary_candidate,read_ordinary_reference,read_ordinary_candidate)
    assert reference['observer']['selected_steps']==candidate['observer']['selected_steps']
    selected=reference['observer']['selected_steps'];assert len(selected)==1
    step=selected[0];position=keys.index(selected_key)
    assert original['steps'][step-1]['comparison']['rows'][position]['all_exact'] is False,'Only trace an affected ordinary row/step'
    r=layers(reference,lambda n:read_reference(None,n));c=layers(candidate,read_candidate)
    assert set(r)=={(step,selected_key)} and set(c)=={(step,key) for key in keys}
    a,b=r[step,selected_key],c[step,selected_key];assert a['consumed_token']==b['consumed_token']
    diff={str(layer):difference(a['boundaries'][layer],b['boundaries'][layer],active_words=5120) for layer in LAYERS}
    return dict(status='affected_row_all64_layers_compared_after_complete_primary_parity',selected_key=selected_key,physical_row=position,step=step,primary_parity=parity,
                boundaries=diff,first_differing_full_T4_layer=next((i for i in LAYERS if not diff[str(i)]['all_bytes_exact']),None),
                first_differing_active_layer=next((i for i in LAYERS if diff[str(i)]['active_different_words']),None),
                ordinary_cross_profile_state_observation=original,numerical_tolerance_applied=False,original_full4k_admission_still_failed=True,MTP_quality_HTTP_or_performance_qualified=False,
                scope='One affectedrow atone proved ordinarydriftstep. All64completeT4layeroutputs retained;5120active/15360inactive words separated. Everyordinary same-profilecomplete128componentstate/finitehidden-normvector snapshotafter1..7andall8IDs mustmatchfor selectedB1andallfourB4rows before interpretation. OtherB1layeroutputs unobserved. Firstdifferentboundary isnotuniqueoperatorcausality or numerical/performanceadmission.')
