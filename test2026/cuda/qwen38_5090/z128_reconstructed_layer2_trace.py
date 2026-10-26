"""Native stderr attribution of the explicit factory to the new Garnet plugin."""
import json
from .z128_reconstructed_plugin_init_contract import PREFIX as PLUGIN,events as plugin_events,replay as plugin_replay
from .z128_reconstruct import PREFIX as FACTORY,events as factory_events,replay as factory_replay
def replay(raw,enabled):
    if isinstance(raw,bytes):raw=raw.decode('utf8',errors='replace')
    plugins=plugin_events(raw);factories=factory_events(raw)
    if not enabled:
        assert plugins==factories==[]
        return {'status':'actual_reconstructed_factory_and_plugin_tracing_unset_silent','events':0}
    p=plugin_replay(plugins);q=factory_replay(raw);active={};links={};open_factories=set()
    for line in raw.splitlines():
        if line.startswith(PLUGIN):
            row=json.loads(line[len(PLUGIN):])
            if row['phase']=='begin':active[row['id']]=row
            else:
                assert row['id'] in active
                for identity,link in links.items():
                    if link['plugin_begin']['id']==row['id']:
                        assert identity not in open_factories;link['plugin_end']=row
                del active[row['id']]
        elif line.startswith(FACTORY):
            row=json.loads(line[len(FACTORY):])
            if row['event']=='begin':
                assert len(active)==1 and next(iter(active.values()))['route']=='fp8_z128_reconstructed','Ambiguous concurrent plugin attribution'
                assert row['id'] not in links;links[row['id']]={'plugin_begin':next(iter(active.values()))};open_factories.add(row['id'])
            elif row['event']=='end':assert row['id'] in open_factories;open_factories.remove(row['id'])
            else:assert row['id'] in open_factories
    assert not active and not open_factories and set(links)==set(range(1,q['factory_calls']+1))
    for call in q['calls']:
        link=links[call['id']];a=link['plugin_begin'];b=link['plugin_end'];assert a['instance']==b['instance'] and a['route']==b['route']=='fp8_z128_reconstructed' and b['factory_called']
        assert (a['rows'],a['columns'],a['policy'])==(6144,5120,0x52313238)
        if call['outcome'] in ('created','cached'):assert b['result']==0 and b['context_present'] and not b['exception']
        elif call['outcome']=='empty':assert b['result']==1 and not b['context_present'] and not b['exception']
        else:assert b['result']==1 and b['exception']
    return {'status':'actual_reconstructed_factory_and_enclosing_plugin_replayed','plugin':p,'factory':q,'native_stderr_attribution':links,'model_or_performance_qualified':False}
