"""Actual V component route proof, inherited live-SDK checks, no timing claim."""
import json,re
from .b4_b1_short_m16_gate import FACTORY_APIS

def route_trace(raw):
    if isinstance(raw,bytes):raw=raw.decode('utf8',errors='replace')
    factory=[json.loads(line.split('GARNET_VM16_FACTORY ',1)[1]) for line in raw.splitlines() if line.startswith('GARNET_VM16_FACTORY ')]
    initialization=[json.loads(line.split('GARNET_MODELOPT_INIT ',1)[1]) for line in raw.splitlines() if line.startswith('GARNET_MODELOPT_INIT ')]
    groups={}
    for row in factory:assert set(row)=={'id','seq','event','name','a','b','c','opaque'};groups.setdefault(row['id'],[]).append(row)
    assert len(groups)>=1;created=[];cached=0
    for rows in groups.values():
        assert [r['seq'] for r in rows]==list(range(1,len(rows)+1)) and rows[0]['event']=='begin' and rows[-1]['event']=='end'
        assert rows[-1]['name'] in ('created','cached')
        apis=[r for r in rows if r['event']=='api']
        if rows[-1]['name']=='cached':assert len(apis)==3 and [r['name'] for r in apis]==['cudaGetDevice','cudaGetDeviceProperties','cublasLtGetVersion'] and [r['a'] for r in apis]==[0,0,130400];cached+=1;continue
        assert [r['name'] for r in apis]==FACTORY_APIS and all(r['a']==(130400 if r['name']=='cublasLtGetVersion' else 0) for r in apis)
        configs=[r for r in rows if r['event']=='configuration'];assert [(r['name'],r['a'],r['b'],r['c']) for r in configs]==[(n,v,4,v) for n,v in zip(('ID','TILE_ID','STAGES_ID','SPLITK_NUM','REDUCTION_SCHEME','CTA_SWIZZLING','CUSTOM_OPTION'),(35,13,17,3,2,0,1))]
        checks=[r for r in rows if r['event']=='check'];assert [r['name'] for r in checks]==['M4','M16'] and all(r['a']==0 and 0<=r['b']<=4<<20 for r in checks)
        selected=[r for r in rows if r['event']=='selection'];assert len(selected)==1 and selected[0]['name']=='observed_M4_control' and re.fullmatch('[0-9a-f]{128}',selected[0]['opaque'])
        eligible=[r for r in rows if r['event']=='candidate' and r['a']==selected[0]['a']];assert len(eligible)==1 and eligible[0]['b']==0 and eligible[0]['opaque']==selected[0]['opaque'];created.append(dict(configuration=[r['a'] for r in configs],checks=checks,opaque_live_SDK_selection_preserved=True))
    routes=[r for r in initialization if r['route']=='fp8_v_m16'];assert routes and len(routes)%2==0;pairs={}
    for row in routes:pairs.setdefault(row['id'],[]).append(row)
    counts={1024:0}
    for rows in pairs.values():
        assert len(rows)==2 and [r['phase'] for r in rows]==['begin','end'];a,b=rows;assert a['instance']==b['instance'] and a['rows']==b['rows'] and a['columns']==b['columns']==5120 and a['policy']==b['policy']==16 and b['rows'] in counts and b['context_present'] and not b['exception'] and b['result']==0;counts[b['rows']]+=1
    assert created and all(n>=2 for n in counts.values())
    return dict(status='actual_B4T4_V_M16_route_and_live_SDK_selection_replayed',created_contexts=len(created),cached_factory_calls=cached,initializations_by_output_rows={str(n):v for n,v in counts.items()},created=created,all_seven_attributes_checked=True,original_4MiB_workspace_preserved=True)

def assert_disabled(raw):
    if isinstance(raw,bytes):raw=raw.decode("utf8",errors="replace")
    assert not any(line.startswith("GARNET_VM16_FACTORY ") for line in raw.splitlines())
    assert not any(json.loads(line.split("GARNET_MODELOPT_INIT ",1)[1])["route"]=="fp8_v_m16" for line in raw.splitlines() if line.startswith("GARNET_MODELOPT_INIT "))
    return {"status":"V_M16_route_absent_in_reference_or_off"}
