import copy,json,unittest
from . import test_v_m16_component_trace as component_fixtures
from .b4_v_m16_short_route import route_trace

class FullRoutes(unittest.TestCase):
    def fixture(self):
        original,_=component_fixtures.Trace().fixture();lines=[]
        for context in (1,2):
            rows=copy.deepcopy(original)
            for row in rows:row['id']=context
            lines += ['GARNET_GDNM16_FACTORY '+json.dumps(row) for row in rows]
        lines += ['GARNET_VM16_FACTORY '+json.dumps(row) for row in original]
        for i in range(112):
            route='fp8_gdn_m16' if i<96 else 'fp8_v_m16';n=(10240 if i<48 else 6144) if i<96 else 1024
            for phase in ('begin','end'):lines.append('GARNET_MODELOPT_INIT '+json.dumps(dict(id=i+1,phase=phase,route=route,instance=str(i),rows=n,columns=5120,policy=16,context_present=phase=='end',factory_called=phase=='end',result=0 if phase=='end' else -1,exception=False)))
        return '\n'.join(lines)
    def test_original_GDN_and_all_sixteen_V_routes_required(self):
        q=route_trace(self.fixture());self.assertEqual(q['GDN']['initializations_by_output_rows'],{'10240':48,'6144':48});self.assertEqual(q['V']['initializations_by_output_rows'],{'1024':16});self.assertFalse(q['model_or_quality_or_performance_qualified'])
    def test_component_two_V_routes_insufficient_for_complete_model(self):
        raw=self.fixture();lines=[]
        for line in raw.splitlines():
            if line.startswith('GARNET_MODELOPT_INIT '):
                row=json.loads(line.split(' ',1)[1])
                if row['route']=='fp8_v_m16' and row['id']>98:continue
            lines.append(line)
        with self.assertRaises(AssertionError):route_trace('\n'.join(lines))
    def test_original_GDN_route_cannot_be_waived_by_V_success(self):
        raw=self.fixture()
        for bad in ('\n'.join(line for line in raw.splitlines() if not line.startswith('GARNET_GDNM16_FACTORY ')),raw.replace('"policy": 16','"policy": 4',1)):
            with self.assertRaises(AssertionError):route_trace(bad)
if __name__=='__main__':unittest.main()
