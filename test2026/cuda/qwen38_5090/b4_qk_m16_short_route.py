"""Require old GDN/V routes plus live Q/K configurations for complete B4 decode."""
import json,re
from .b4_v_m16_short_route import route_trace as previous_route
from .v_m16_component_trace import assert_disabled as previous_disabled
from .b4_b1_short_m16_gate import FACTORY_APIS
CONFIGS={1024:(35,13,17,3,2,0,1),12288:(35,18,16,7,2,1,2)}
ATTRS=('ID','TILE_ID','STAGES_ID','SPLITK_NUM','REDUCTION_SCHEME','CTA_SWIZZLING','CUSTOM_OPTION')

def route_trace(raw):
    if isinstance(raw,bytes):raw=raw.decode('utf8',errors='replace')
    previous=previous_route(raw);groups={};initializations=[]
    for line in raw.splitlines():
        if line.startswith('GARNET_QKM16_FACTORY '):
            row=json.loads(line.split(' ',1)[1]);assert set(row)=={'id','seq','event','name','a','b','c','opaque'};groups.setdefault(row['id'],[]).append(row)
        if line.startswith('GARNET_MODELOPT_INIT '):
            row=json.loads(line.split(' ',1)[1])
            if row['route']=='fp8_qk_m16':initializations.append(row)
    assert groups;created=[];cached=0;config_geometries=set()
    for rows in groups.values():
        assert [r['seq'] for r in rows]==list(range(1,len(rows)+1)) and rows[0]['event']=='begin' and rows[-1]['event']=='end' and rows[-1]['name'] in ('created','cached')
        apis=[r for r in rows if r['event']=='api']
        if rows[-1]['name']=='cached':
            assert [r['name'] for r in apis]==['cudaGetDevice','cudaGetDeviceProperties','cublasLtGetVersion'] and [r['a'] for r in apis]==[0,0,130400];cached+=1;continue
        assert [r['name'] for r in apis]==FACTORY_APIS and all(r['a']==(130400 if r['name']=='cublasLtGetVersion' else 0) for r in apis)
        configs=[r for r in rows if r['event']=='configuration'];observed=tuple(r['a'] for r in configs);matched=[n for n,c in CONFIGS.items() if c==observed];assert len(matched)==1;n=matched[0]
        assert [(r['name'],r['a'],r['b'],r['c']) for r in configs]==[(name,v,4,v) for name,v in zip(ATTRS,CONFIGS[n])]
        checks=[r for r in rows if r['event']=='check'];assert [r['name'] for r in checks]==['M4','M16'] and all(r['a']==0 and 0<=r['b']<=4<<20 for r in checks)
        selected=[r for r in rows if r['event']=='selection'];assert len(selected)==1 and selected[0]['name']=='observed_M4_control' and re.fullmatch('[0-9a-f]{128}',selected[0]['opaque'])
        eligible=[r for r in rows if r['event']=='candidate' and r['a']==selected[0]['a']];assert len(eligible)==1 and eligible[0]['b']==0 and eligible[0]['opaque']==selected[0]['opaque']
        config_geometries.add(n);created.append(dict(configuration=list(observed),geometry_inferred_from_configuration=n,checks=checks,opaque_live_SDK_selection_preserved=True))
    assert config_geometries==set(CONFIGS)
    pairs={};counts={1024:0,12288:0}
    for row in initializations:pairs.setdefault(row['id'],[]).append(row)
    for rows in pairs.values():
        assert len(rows)==2 and [r['phase'] for r in rows]==['begin','end'];a,b=rows;assert a['instance']==b['instance'] and a['rows']==b['rows'] and a['columns']==b['columns']==5120 and a['policy']==b['policy']==16 and b['rows'] in counts and b['context_present'] and not b['exception'] and b['result']==0;counts[b['rows']]+=1
    assert all(v>=16 for v in counts.values()),'All sixteen full-attention Q and K projections required'
    return dict(status='actual_B4_GDN_V_QK_M16_full_model_routes_replayed',previous_GDN_and_V=previous,QK=dict(created_contexts=len(created),cached_factory_calls=cached,created=created,initializations_by_output_rows={str(n):v for n,v in counts.items()},all_seven_attributes_checked=True,original_4MiB_workspace_preserved=True),model_or_quality_or_performance_qualified=False,scope='Factory geometry is inferred from the unique configuration tuple and pinned code/CPU contracts; actual plugin initialization records supply N/K. This route proof alone is not actual projection or whole-model numerical admission.')

def assert_disabled(raw):
    previous_disabled(raw)
    if isinstance(raw,bytes):raw=raw.decode('utf8',errors='replace')
    assert not any(line.startswith('GARNET_QKM16_FACTORY ') for line in raw.splitlines())
    assert not any(json.loads(line.split(' ',1)[1])['route']=='fp8_qk_m16' for line in raw.splitlines() if line.startswith('GARNET_MODELOPT_INIT '))
    return dict(status='V_and_QK_M16_routes_absent_in_reference')
