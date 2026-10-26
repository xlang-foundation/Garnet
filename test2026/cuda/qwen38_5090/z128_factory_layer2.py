"""Join native stderr factory records to the actual enclosing Z128 plugin initialize call."""
import json
from .modelopt_init_trace_contract import PREFIX as PLUGIN,events as plugin_events
from .z128_factory_trace import PREFIX as FACTORY,replay,events

def joined_replay(raw,enabled):
    if isinstance(raw,bytes):raw=raw.decode('utf8',errors='replace')
    if not enabled:
        assert events(raw)==[];return {'status':'factory_trace_unset_silent','events':0,'actual_failure_guard_observed':False}
    q=replay(raw);active={};links={};seen={};ambiguous=False
    for line in raw.splitlines():
        if line.startswith(PLUGIN):
            row=plugin_events(line)[0]
            if row['phase']=='begin':active[row['id']]=row
            else:
                assert row['id'] in active
                for identity in list(seen):
                    if links[identity]['plugin_begin']['id']==row['id']:
                        assert seen[identity]=='ended';links[identity]['plugin_end']=row;del seen[identity]
                del active[row['id']]
        elif line.startswith(FACTORY):
            row=json.loads(line[len(FACTORY):])
            if row['event']=='begin':
                # Without thread IDs only a single enclosing plugin call is unambiguous.
                if len(active)!=1 or next(iter(active.values()))['route']!='fp8_z128':ambiguous=True
                assert not ambiguous,'Interleaved native plugin calls prevent exact factory attribution'
                assert row['id'] not in links;links[row['id']]={'plugin_begin':next(iter(active.values()))};seen[row['id']]='begun'
            elif row['event']=='end':assert seen[row['id']]=='begun';seen[row['id']]='ended'
            else:assert seen[row['id']]=='begun'
    assert not seen and not active and set(links)==set(range(1,q['factory_calls']+1))
    failures=[]
    for call in q['calls']:
        link=links[call['id']];begin=link['plugin_begin'];end=link['plugin_end'];assert end['factory_called'] and begin['route']==end['route']=='fp8_z128' and begin['instance']==end['instance'] and begin['rows']==end['rows']==6144 and begin['columns']==end['columns']==5120
        if call['outcome']=='empty':
            assert end['result']==1 and not end['context_present'] and not end['exception'];failures.append({'factory_id':call['id'],'plugin_id':begin['id'],'instance':begin['instance'],'guard':call['reject'],'SDK_results':call['API_results'],'heuristic_candidates':call['candidates']})
        elif call['outcome'] in ('created','cached'):assert end['result']==0 and end['context_present'] and not end['exception']
        else:assert end['result']==1 and end['exception']
    return {'status':'actual_Garnet_Z128_factory_and_enclosing_plugin_replayed','events':q['events'],'factory':q,'actual_failure_guard_observed':bool(failures),'failures':failures,'plugin_factory_attribution':'native_stderr_single_unambiguous_enclosing_plugin_call','Python_profile_stdout_chronology_used':False,'algorithm_substituted':False,'model_or_quality_or_performance_qualified':False}
