"""Actual initialization-event attribution and unchanged complete ordinary-array controls."""
import math
from .modelopt_init_trace_contract import PREFIX,events,replay as replay_events
from .gdn_chunk_stage_replay import array
from .layer2_ordinary_replay import replay as replay_arrays

BEGIN='GARNET_TRACE_PROFILE_BEGIN '
READY='GARNET_TRACE_PROFILE_READY '
def trace_profiles(raw,enabled,complete):
    if isinstance(raw,bytes):raw=raw.decode('utf8',errors='replace')
    rows=events(raw);started=[];ready=[];current=None;profiles={};ends={}
    for line in raw.splitlines():
        if line.startswith(BEGIN):
            token=int(line[len(BEGIN):]);assert token in (32,128) and token not in started;started.append(token);current=token
        elif line.startswith(READY):
            token=int(line[len(READY):]);assert token==current and token in started and token not in ready;ready.append(token)
        elif line.startswith(PREFIX):
            row=events(line)[0]
            if row['phase']=='begin':profiles[row['id']]=current
            else:ends[row['id']]=current
    assert started in ([],[32],[32,128]) and ready in ([],[32],[32,128]) and set(ready)<=set(started)
    if complete:assert started==ready==[32,128]
    if not enabled:assert rows==[];return {'status':'trace_unset_silent','started_profiles':started,'ready_profiles':ready,'events':0,'actual_failed_plugin_route_observed':False}
    if not rows:return {'status':'no_plugin_initialize_events_before_failure','started_profiles':started,'ready_profiles':ready,'events':0,'actual_failed_plugin_route_observed':False}
    result=replay_events(rows);assert set(profiles)==set(ends)
    failures=[dict(row,begin_profile=profiles[row['id']],end_profile=ends[row['id']]) for row in result['failures']]
    if complete:assert not failures
    return {'status':'actual_layer2_initialize_trace_replayed','started_profiles':started,'ready_profiles':ready,'events':len(rows),'trace_replay':result,'call_begin_profiles':{str(k):v for k,v in profiles.items()},'failed_initialize_returns':failures,'actual_failed_plugin_route_observed':bool(failures),'helper_API_failure_cause_established':False}

def compare_profiles(actual,prior,read,oldread,tokens):
    assert array(actual['common_input'],read)==array(prior['common_input'],oldread)
    compared=1
    for token in tokens:
        a=[row for row in actual['cases'] if row['tokens']==token];b=[row for row in prior['cases'] if row['tokens']==token];assert len(a)==len(b)==1;a=a[0];b=b[0]
        for name in ('layer','tokens','diagnostic','layout','input_lineage'):assert a[name]==b[name]
        assert array(a['active'],read)==array(b['active'],oldread);compared+=1
        assert len(a['repeats'])==len(b['repeats'])==2
        for sequence,old in zip(a['repeats'],b['repeats']):
            assert len(sequence)==len(old)==128//token
            for current,earlier in zip(sequence,old):
                assert current['start']==earlier['start'] and current['repeat']==earlier['repeat'] and set(current['arrays'])==set(earlier['arrays'])=={'primary'}
                for name in ('input_hidden','input_history','input_state','input_hidden_after','input_history_after','input_state_after'):
                    assert array(current[name],read)==array(earlier[name],oldread);compared+=1
                assert array(current['arrays']['primary'],read)==array(earlier['arrays']['primary'],oldread);compared+=1
    return {'complete_finite_entries_compared':compared,'profiles':list(tokens),'all_raw_bytes_exact':True}

def stable_array_replay(report,read,common):
    result=replay_arrays(report,read,common)
    a,b=report['cases'];assert (a['tokens'],b['tokens'])==(32,128)
    import struct
    left=b''.join(array(chunk['arrays']['primary'],read)[:32*5120*4] for chunk in a['repeats'][0]);right=array(b['repeats'][0][0]['arrays']['primary'],read)[:128*5120*4]
    x=struct.unpack('<655360f',left);y=struct.unpack('<655360f',right);norm=math.fsum(float(v)*float(v) for v in x);delta=math.fsum((float(v)-float(w))**2 for v,w in zip(x,y))
    result['hidden_relative_l2_difference']=math.sqrt(delta/norm) if norm else (0.0 if delta==0 else None);result['norm_method']='Python_float64_fsum_of_saved_F32_arrays'
    return result
