"""Reconcile actual current-native DOTs with captures, output and ownership gates."""
import datetime
import hashlib
import json
from pathlib import Path
import re
import sys
from qwen35_capture_dot import parse_dot
root=Path(sys.argv[1]).resolve();stem=root/'qwen35-capture-topology-v1'
output=Path(str(stem)+'-summary.json');assert not output.exists()
sources={}
def read(path):
    blob=Path(path).read_bytes();sources[str(path)]=hashlib.sha256(blob).hexdigest()
    return json.loads(blob.decode('utf-8-sig'))
def timestamp(s):return datetime.datetime.fromisoformat(s.replace('Z','+00:00'))
def graph_stats(value,path='owned_memory'):
    result={}
    if isinstance(value,dict):
        if 'graph_captures' in value:result[path]=value
        for key,item in value.items():result.update(graph_stats(item,path+'.'+key))
    elif isinstance(value,list):
        for index,item in enumerate(value):result.update(graph_stats(item,path+f'[{index}]'))
    return result
identity_path=Path(str(stem)+'-before-run-identity.json')
identity=read(identity_path);epoch=sources[str(identity_path)]
observer_path=Path(str(stem)+'-processes.jsonl')
blob=observer_path.read_bytes();sources[str(observer_path)]=hashlib.sha256(blob).hexdigest()
observer=[json.loads(x) for x in blob.decode('utf-8-sig').splitlines()]
assert observer[0]['event']=='observer_started' and observer[-1]['event']=='stopped'
cases=[];previous_end=None
for size,batch in [('2b',1),('4b',8)]:
    tag=root/f'profile-qwen35-capture-topology-{size}-b{batch}-v1'
    exit=read(Path(str(tag)+'-exit.json'));assert exit['exit_code']==0
    assert exit['execution_identity_sha256'].lower()==epoch
    assert exit['garnet_dll_sha256'].lower()==identity['garnet_binary_sha256'].lower()
    assert exit['gpu_uuid']=='GPU-472bb21a-0ca8-9d60-47a0-c527bb3d2d81'
    rows=read(Path(str(tag)+'-requests.json'))
    reference=read(root/f'matched512-garnet-{size}-b{batch}-study-shared-mtp-v1-a1.json')
    assert len(rows)==len(reference)==28
    for index,(row,ref) in enumerate(zip(rows,reference)):
        for field in ('profile','native_build_policy','input_sha256','prompt_token_ids','generated_token_ids',
                      'sampled_tokens_including_stop','finish_reasons','stop_token_ids','cache_policy','speculative_metrics','visual_tokens_per_row'):
            assert row[field]==ref[field],(size,index,field)
        assert row['owned_memory']['owned_model_bytes']==ref['owned_memory']['owned_model_bytes']
        assert row['owned_memory']['speculative']['mtp_graph']['contexts']==ref['owned_memory']['speculative']['mtp_graph']['contexts']==1
        metadata=row['capture_topology_diagnostic']
        assert metadata['request_index']==index and not metadata['native_layer_profiler_enabled'] and not metadata['primary_latency_accepted']
        assert not metadata['changed_existing_dot_files_this_request']
        assert row['warmup']==(index<8) and row['vision_compile_ms']==0
    assert read(Path(str(tag)+'-preservation.json'))['same_all28_input_output_stop_profile_speculative_counters']
    stats=graph_stats(rows[-1]['owned_memory']);prior_stats=graph_stats(reference[-1]['owned_memory'])
    assert set(stats)==set(prior_stats) and len(stats)==7
    for role,stat in stats.items():
        assert stat['graph_captures']==prior_stats[role]['graph_captures']==1
        assert stat['graph_capture_failures']==0 and stat['contexts']==1
    expected_capture_count=sum(x['graph_captures'] for x in stats.values())
    inventory=read(Path(str(tag)+'-requests.capture-inventory.json'))
    assert inventory['all28_completed'] and len(inventory['new_dots'])==expected_capture_count
    log_path=Path(str(tag)+'.log');log=log_path.read_text(encoding='utf-8',errors='replace')
    sources[str(log_path)]=hashlib.sha256(log_path.read_bytes()).hexdigest()
    capture_logs=[];pending=[]
    for line in log.splitlines():
        node=re.search(r'TRT_CAPTURE_NODES depth=(\d+) nodes=(\d+) allocations=(\d+) frees=(\d+) children=(\d+) engine=(.+)',line)
        if node:
            d,n,a,f,c,engine=node.groups()
            pending.append({'depth':int(d),'nodes':int(n),'allocations':int(a),'frees':int(f),'children':int(c),'engine':engine})
        dot=re.search(r'TRT_CAPTURE_DOT status=(\d+) path=(.+)',line)
        if dot:
            status,path=dot.groups();assert int(status)==0 and pending
            assert sum(x['depth']==0 for x in pending)==1
            assert len({x['engine'] for x in pending})==1
            capture_logs.append({'path':Path(path),'nodes':pending});pending=[]
    assert not pending and len(capture_logs)==expected_capture_count
    assert len({str(x['path']).lower() for x in capture_logs})==expected_capture_count,'DOT path overwritten during repeated capture'
    lookup={str(x['path']).lower():x for x in capture_logs};topologies=[]
    for entry in inventory['new_dots']:
        path=Path(entry['file']);raw=path.read_bytes();sha=hashlib.sha256(raw).hexdigest()
        assert len(raw)==entry['bytes'] and sha==entry['sha256']
        sources[str(path)]=sha
        trace=lookup[str(path).lower()];parsed=parse_dot(raw.decode())
        assert parsed['node_count']==sum(x['nodes'] for x in trace['nodes'])
        topologies.append({'file':str(path),'sha256':sha,'capture_trace':trace['nodes'],'topology':parsed})
    start,end=timestamp(exit['started_utc']),timestamp(exit['ended_utc'])
    assert previous_end is None or previous_end<=start;previous_end=end
    assert timestamp(observer[0]['utc'])<=start<end<=timestamp(observer[-1]['utc'])
    pid=rows[0]['process_id'];assert all(r['process_id']==pid for r in rows)
    snaps=[s for s in observer if s['event']=='snapshot' and timestamp(s['begin_utc'])<=end and timestamp(s['end_utc'])>=start]
    assert any(p['pid']==pid for s in snaps for p in s['processes'])
    foreign=[{'snapshot':s['begin_utc'],'process':p} for s in snaps for p in s['processes'] if p['pid']!=pid]
    errors=[x for x in observer if x['event']=='observer_failure' or x['event']=='process_metadata_unavailable' and start<=timestamp(x['observed_utc'])<=end]
    cases.append({'model':size,'batch':batch,'pid':pid,'all28_work_owner_controls_passed':True,
        'retained_graph_capture_counts':{k:x['graph_captures'] for k,x in stats.items()},
        'all_retained_graph_captures_covered':True,'topologies':topologies,
        'all_node_types_permitted_in_conditional_body':all(x['topology']['all_node_types_permitted_in_conditional_body'] for x in topologies),
        'sampled_foreign_processes':foreign,'observer_metadata_errors':errors,'process_observation_qualified':not foreign and not errors})
result={'all56_native_work_owner_and_capture_coverage_controls_passed':True,'cases':cases,
    'execution_identity_sha256':epoch,'garnet_dll_sha256':identity['garnet_binary_sha256'],
    'maximum_snapshot_gap_seconds':observer[-1]['maximum_snapshot_gap_seconds'],
    'all_node_types_permitted_in_conditional_body':all(c['all_node_types_permitted_in_conditional_body'] for c in cases),
    'process_observation_qualified':all(c['process_observation_qualified'] for c in cases),
    'successful_conditional_composition_proven':False,'owners_retained_through_outer_completion_proven':False,
    'kernel_device_launch_absence_proven':False,'copy_operand_accessibility_proven':False,
    'primary_latency_accepted':False,'memory_accepted':False,'production_modified':False,
    'sources_sha256':sources,'scope':'All current retained real-model capture topology coverage with complete original request/work/owned-byte preservation. Node type screen only; no actual child-graph export or GPU speculative loop, model speed/memory/quality or2x acceptance.'}
with output.open('x') as stream:json.dump(result,stream,indent=2)
print('CAPTURE_TOPOLOGY_FULL_AUDIT_COMPLETE',len(cases),'NODE_TYPE_SCREEN',result['all_node_types_permitted_in_conditional_body'])
