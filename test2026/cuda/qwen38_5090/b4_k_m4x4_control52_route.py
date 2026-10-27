"""Live SDK initialization evidence for a single flag intervention; no speed claim."""
import json,re
from .b4_b1_short_m16_gate import FACTORY_APIS
from .b4_v_m16_short_route import route_trace as old_gdn_v
from .b4_qk_m16_short_route import route_trace as old_qk
ATTRS=('ID','TILE_ID','STAGES_ID','SPLITK_NUM','REDUCTION_SCHEME','CTA_SWIZZLING','CUSTOM_OPTION')

def _contexts(raw,prefix,route,configs,check_names,policy):
    groups={};initializations=[]
    for line in raw.splitlines():
        if line.startswith(prefix):
            r=json.loads(line.split(' ',1)[1]);assert set(r)=={'id','seq','event','name','a','b','c','opaque'};groups.setdefault(r['id'],[]).append(r)
        if line.startswith('GARNET_MODELOPT_INIT '):
            r=json.loads(line.split(' ',1)[1])
            if r['route']==route:initializations.append(r)
    assert groups;created=[];cached=0;geometries=set()
    for rows in groups.values():
        assert [r['seq'] for r in rows]==list(range(1,len(rows)+1)) and rows[0]['event']=='begin' and rows[-1]['event']=='end'
        assert rows[-1]['name'] in ('created','cached');apis=[r for r in rows if r['event']=='api']
        if rows[-1]['name']=='cached':
            assert [r['name'] for r in apis]==FACTORY_APIS[:3] and [r['a'] for r in apis]==[0,0,130400];cached+=1;continue
        assert [r['name'] for r in apis]==FACTORY_APIS and all(r['a']==(130400 if r['name']=='cublasLtGetVersion' else 0) for r in apis)
        config=[r for r in rows if r['event']=='configuration'];values=tuple(r['a'] for r in config);matches=[n for n,c in configs.items() if c==values];assert len(matches)==1;n=matches[0]
        assert [(r['name'],r['a'],r['b'],r['c']) for r in config]==[(name,v,4,v) for name,v in zip(ATTRS,configs[n])]
        checks=[r for r in rows if r['event']=='check'];assert [r['name'] for r in checks]==list(check_names) and all(r['a']==0 and 0<=r['b']<=4<<20 for r in checks)
        selected=[r for r in rows if r['event']=='selection'];assert len(selected)==1 and selected[0]['name']=='observed_M4_control' and re.fullmatch('[0-9a-f]{128}',selected[0]['opaque'])
        eligible=[r for r in rows if r['event']=='candidate' and r['a']==selected[0]['a']];assert len(eligible)==1 and eligible[0]['b']==0 and eligible[0]['opaque']==selected[0]['opaque']
        geometries.add(n);created.append(dict(geometry_inferred_from_configuration=n,configuration=list(values),checks=checks,opaque_live_SDK_selection_preserved=True))
    assert geometries==set(configs);pairs={};counts={n:0 for n in configs}
    for row in initializations:pairs.setdefault(row['id'],[]).append(row)
    for rows in pairs.values():
        assert len(rows)==2 and [r['phase'] for r in rows]==['begin','end'];a,b=rows
        assert a['instance']==b['instance'] and a['rows']==b['rows'] and a['columns']==b['columns']==5120 and a['policy']==b['policy']==policy and b['rows'] in counts
        assert b['context_present'] and not b['exception'] and b['result']==0;counts[b['rows']]+=1
    assert all(n>=16 for n in counts.values())
    return dict(created_contexts=len(created),cached_factory_calls=cached,created=created,initializations_by_output_rows={str(n):v for n,v in counts.items()},all_seven_attributes_checked=True,original_4MiB_workspace_preserved=True)

def route_trace(raw,policy):
    assert policy in ('off','on')
    if isinstance(raw,bytes):raw=raw.decode('utf8',errors='replace')
    if policy=='off':
        assert not any(line.startswith('GARNET_KM4X4_FACTORY ') for line in raw.splitlines())
        assert not any(json.loads(line.split(' ',1)[1])['route']=='fp8_k_m4x4' for line in raw.splitlines() if line.startswith('GARNET_MODELOPT_INIT '))
        return dict(status='same_DLL_flagoff_original_GDN_V_QK_routes_replayed',original=old_qk(raw),new_K_route_absent=True,model_or_performance_qualified=False)
    previous=old_gdn_v(raw)
    query=_contexts(raw,'GARNET_QKM16_FACTORY ','fp8_qk_m16',{12288:(35,18,16,7,2,1,2)},('M4','M16'),16)
    key=_contexts(raw,'GARNET_KM4X4_FACTORY ','fp8_k_m4x4',{1024:(35,13,17,3,2,0,1)},('M4','M4_actual'),4)
    return dict(status='same_DLL_flagon_GDN_V_Qonly_M16_and_K_fourM4_routes_replayed',previous_GDN_V=previous,Q_only=query,K_fourM4=key,model_or_performance_qualified=False,scope='Live factory/initialize evidence plus pinned native source and CPU run contracts establish selected route eligibility; geometry is inferred from configuration and code. Actual numerical admission is separate.')
