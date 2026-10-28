import copy,json,unittest
from . import test_b4_v_m16_short_route as previous_fixtures
from . import test_v_m16_component_trace as component_fixtures
from .b4_qk_m16_short_route import route_trace,assert_disabled,CONFIGS
class FullQKRoutes(unittest.TestCase):
    def fixture(self):
        lines=previous_fixtures.FullRoutes().fixture().splitlines();factory,_=component_fixtures.Trace().fixture()
        for context,(n,configuration) in enumerate(CONFIGS.items(),1):
            rows=copy.deepcopy(factory);configs=[r for r in rows if r['event']=='configuration']
            for row,v in zip(configs,configuration):row['a']=row['c']=v
            for row in rows:row['id']=context
            lines.extend('GARNET_QKM16_FACTORY '+json.dumps(row) for row in rows)
            for i in range(16):
                for phase in ('begin','end'):lines.append('GARNET_MODELOPT_INIT '+json.dumps(dict(id=200+context*16+i,phase=phase,route='fp8_qk_m16',instance=str(i),rows=n,columns=5120,policy=16,context_present=phase=='end',factory_called=phase=='end',result=0 if phase=='end' else -1,exception=False)))
        return '\n'.join(lines)
    def test_both_geometry_specific_full_configs_and_all_QK_routes(self):
        q=route_trace(self.fixture());self.assertEqual(q['QK']['initializations_by_output_rows'],{'1024':16,'12288':16});self.assertEqual(q['QK']['created_contexts'],2);self.assertFalse(q['model_or_quality_or_performance_qualified'])
    def test_all_seven_attributes_live_opaque_API_workspace_and_sequence(self):
        for mutation in ('configuration','bytes','API','workspace','opaque','sequence'):
            lines=self.fixture().splitlines();rows=[(i,json.loads(line.split(' ',1)[1])) for i,line in enumerate(lines) if line.startswith('GARNET_QKM16_FACTORY ')]
            if mutation=='configuration':index,row=next((i,r) for i,r in rows if r['id']==2 and r['event']=='configuration' and r['name']=='TILE_ID');row['a']=13
            elif mutation=='bytes':index,row=next((i,r) for i,r in rows if r['event']=='configuration');row['b']=3
            elif mutation=='API':index,row=next((i,r) for i,r in rows if r['event']=='api');row['a']=7
            elif mutation=='workspace':index,row=next((i,r) for i,r in rows if r['event']=='check');row['b']=(4<<20)+1
            elif mutation=='opaque':index,row=next((i,r) for i,r in rows if r['event']=='selection');row['opaque']='cd'*64
            else:index,row=rows[-1];row['seq']+=1
            lines[index]='GARNET_QKM16_FACTORY '+json.dumps(row)
            with self.assertRaises(AssertionError):route_trace('\n'.join(lines))
    def test_old_GDN_V_and_all_sixteen_Q_K_initializations_remain_required(self):
        for mutation in ('GDN','V','oneQ','oneK','policy','geometry'):
            lines=[];changed=False
            for line in self.fixture().splitlines():
                if mutation in ('GDN','V') and line.startswith('GARNET_'+('GDNM16' if mutation=='GDN' else 'VM16')+'_FACTORY '):continue
                if line.startswith('GARNET_MODELOPT_INIT '):
                    row=json.loads(line.split(' ',1)[1])
                    if row['route']=='fp8_qk_m16' and mutation in ('oneQ','oneK') and row['id']==(232 if mutation=='oneQ' else 216):continue
                    if row['route']=='fp8_qk_m16' and not changed and mutation in ('policy','geometry'):
                        row['policy' if mutation=='policy' else 'rows']=4 if mutation=='policy' else 6144;line='GARNET_MODELOPT_INIT '+json.dumps(row);changed=True
                lines.append(line)
            with self.assertRaises(AssertionError):route_trace('\n'.join(lines))
    def test_reference_has_no_V_or_QK_route(self):
        self.assertEqual(assert_disabled('ordinary reference')['status'],'V_and_QK_M16_routes_absent_in_reference')
        for prefix in ('GARNET_VM16_FACTORY ','GARNET_QKM16_FACTORY '):
            with self.assertRaises(AssertionError):assert_disabled(prefix+'{}')
if __name__=='__main__':unittest.main()
