import copy,json,unittest
from . import test_b4_qk_m16_short_route as old_fixtures
from .b4_k_m4x4_control52_route import route_trace

def fixture(policy):
    raw=old_fixtures.FullQKRoutes().fixture()
    if policy=='off':return raw
    rows=[]
    for line in raw.splitlines():
        if line.startswith('GARNET_QKM16_FACTORY '):
            r=json.loads(line.split(' ',1)[1])
            if r['id']==1:
                if r['event']=='check' and r['name']=='M16':r['name']='M4_actual'
                line='GARNET_KM4X4_FACTORY '+json.dumps(r)
        elif line.startswith('GARNET_MODELOPT_INIT '):
            r=json.loads(line.split(' ',1)[1])
            if r['route']=='fp8_qk_m16' and r['rows']==1024:r['route']='fp8_k_m4x4';r['policy']=4;line='GARNET_MODELOPT_INIT '+json.dumps(r)
        rows.append(line)
    return '\n'.join(rows)

class Routes(unittest.TestCase):
    def test_off_requires_all_original_routes_and_no_new_route(self):
        self.assertTrue(route_trace(fixture('off'),'off')['new_K_route_absent'])
        with self.assertRaises(AssertionError):route_trace(fixture('on'),'off')
    def test_on_requires_query_only_old_route_and_all_new_key_routes(self):
        q=route_trace(fixture('on'),'on');self.assertEqual(q['Q_only']['initializations_by_output_rows'],{'12288':16});self.assertEqual(q['K_fourM4']['initializations_by_output_rows'],{'1024':16});self.assertFalse(q['model_or_performance_qualified'])
    def test_wrong_config_bytes_api_workspace_opaque_sequence_and_actual_M16_rejected(self):
        for mutation in ('configuration','bytes','API','workspace','opaque','sequence','M16'):
            rows=fixture('on').splitlines();parsed=[(i,json.loads(s.split(' ',1)[1])) for i,s in enumerate(rows) if s.startswith('GARNET_KM4X4_FACTORY ')]
            if mutation in ('configuration','bytes'):i,r=next((i,r) for i,r in parsed if r['event']=='configuration');r['a' if mutation=='configuration' else 'b']+=1
            elif mutation=='API':i,r=next((i,r) for i,r in parsed if r['event']=='api');r['a']=7
            elif mutation in ('workspace','M16'):i,r=next((i,r) for i,r in parsed if r['event']=='check' and r['name']=='M4_actual');r['b' if mutation=='workspace' else 'name']=(4<<20)+1 if mutation=='workspace' else 'M16'
            elif mutation=='opaque':i,r=next((i,r) for i,r in parsed if r['event']=='selection');r['opaque']='ef'*64
            else:i,r=parsed[-1];r['seq']+=1
            rows[i]='GARNET_KM4X4_FACTORY '+json.dumps(r)
            with self.assertRaises(AssertionError):route_trace('\n'.join(rows),'on')
    def test_old_GDN_V_Q_and_new_K_all_sixteen_remain_required(self):
        for kind in ('GDN','V','Q','K'):
            rows=[]
            for line in fixture('on').splitlines():
                if kind in ('GDN','V') and line.startswith('GARNET_'+('GDNM16' if kind=='GDN' else 'VM16')+'_FACTORY '):continue
                if line.startswith('GARNET_MODELOPT_INIT '):
                    r=json.loads(line.split(' ',1)[1])
                    if kind in ('Q','K') and r['route']==('fp8_qk_m16' if kind=='Q' else 'fp8_k_m4x4') and r['id']==(232 if kind=='Q' else 216):continue
                rows.append(line)
            with self.assertRaises(AssertionError):route_trace('\n'.join(rows),'on')
    def test_new_policy_must_be4_and_legacy_key_may_not_leak(self):
        for which in ('policy','legacyK'):
            raw=fixture('on')
            if which=='legacyK':
                extra=[]
                for line in fixture('off').splitlines():
                    if line.startswith('GARNET_QKM16_FACTORY ') and json.loads(line.split(' ',1)[1])['id']==1:extra.append(line)
                    elif line.startswith('GARNET_MODELOPT_INIT '):
                        r=json.loads(line.split(' ',1)[1])
                        if r['route']=='fp8_qk_m16' and r['rows']==1024:extra.append(line)
                raw+='\n'+'\n'.join(extra)
            else:raw=raw.replace('"policy": 4','"policy": 16')
            with self.assertRaises(AssertionError):route_trace(raw,'on')
    def test_unknown_intervention_rejected(self):
        with self.assertRaises(AssertionError):route_trace(fixture('off'),'unknown')
if __name__=='__main__':unittest.main()
