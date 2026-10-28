"""Same-build independent process-counter and retained-allocation comparison."""
import argparse,hashlib,json
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('--model',choices=('2b','4b'),required=True);p.add_argument('--batch',type=int,required=True);a=p.parse_args()
root=a.root;prefix=f'memory-combined-{a.model}-b{a.batch}-512-targetserial-aux0-deferred-nativecommit-arena-promptpack-headreuse-joineddecode'
output=root/f'activation-memory-{a.model}-b{a.batch}-pair-summary.json';assert not output.exists(),'Preserve memory pair evidence'
data={};sources=[];binary=set()
for mode,suffix in (('control','activationcontrol'),('candidate','activationarena')):
    tag=prefix+'-'+suffix
    def read(ending):
        path=root/(tag+ending);blob=path.read_bytes();sources.append({'file':path.name,'sha256':hashlib.sha256(blob).hexdigest()});return json.loads(blob)
    exit_data=read('-exit.json');assert exit_data['model_exit_code']==exit_data['monitor_exit_code']==0
    owner=read('-owner.json');binary.add(owner['garnet_dll_sha256'].lower())
    workload=read('-workload-control.json');assert workload['all_six_input_hash_groups_exact'] and workload['all_six_garnet_output_stop_signatures_match_timing'] and workload['same_garnet_profile']
    events=read('-session.json');summary=read('-summary.json');phase=summary['phases']['steady_after'];assert phase['samples']>=3
    steady=next(row for row in events if row['event']=='steady_after_begin');memory=steady['owned_memory']['decoder']
    pool=memory['shared_activation_pool_bytes'];assert bool(pool)==(mode=='candidate')
    assert all(g['user_managed_workspace']==(mode=='candidate') for g in memory['graphs'][:-1])
    if pool:assert all(g['external_workspace_bound'] and g['external_workspace_capacity_bytes']==pool for g in memory['graphs'][:-1])
    data[mode]={'profile':dict(steady['profile']),'sample_count':phase['samples'],
        'warm_process_counter_bytes':{key:phase['bytes'][key]['median'] for key in ('DedicatedUsage','SharedUsage','TotalCommitted')},
        'retained_owned_model_bytes':steady['owned_memory']['owned_model_bytes'],'shared_activation_pool_bytes':pool}
assert len(binary)==1
left=dict(data['control']['profile']);right=dict(data['candidate']['profile'])
assert left.pop('sharePackedActivationStorage',False) is False and right.pop('sharePackedActivationStorage',False) is True
assert left==right,'Other memory profile fields differ'
counter_delta={key:data['control']['warm_process_counter_bytes'][key]-data['candidate']['warm_process_counter_bytes'][key] for key in data['control']['warm_process_counter_bytes']}
result={'model':a.model,'batch':a.batch,'garnet_dll_sha256':next(iter(binary)),'sources':sources,'profiles':data,
    'control_minus_candidate_counter_bytes':counter_delta,
    'control_minus_candidate_retained_owned_bytes':data['control']['retained_owned_model_bytes']-data['candidate']['retained_owned_model_bytes'],
    'scope':'Separate same-DLL control then candidate memory processes with matched profiles/inputs/output signatures. Windows sampled process committed counters and retained accounting are different measures. Not true lifetime peak, physical residency, causal significance, performance or independent quality acceptance.'}
output.write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps({key:value for key,value in result.items() if key not in ('sources','profiles')},indent=2))
