import copy,json,unittest
from .v_m16_component_trace import route_trace,assert_disabled,FACTORY_APIS

class Trace(unittest.TestCase):
    def fixture(self):
        factory=[]
        def row(event,name,a=0,b=0,c=0,opaque=''):
            factory.append(dict(id=1,seq=len(factory)+1,event=event,name=name,a=a,b=b,c=c,opaque=opaque))
        row('begin','factory')
        for api in FACTORY_APIS:row('api',api,130400 if api=='cublasLtGetVersion' else 0)
        row('candidate','heuristic',0,0,100,'ab'*64)
        for name,value in zip(('ID','TILE_ID','STAGES_ID','SPLITK_NUM','REDUCTION_SCHEME','CTA_SWIZZLING','CUSTOM_OPTION'),(35,13,17,3,2,0,1)):row('configuration',name,value,4,value)
        row('selection','observed_M4_control',0,opaque='ab'*64);row('check','M4',0,100);row('check','M16',0,100);row('end','created')
        init=[]
        for i in range(2):
            for phase in ('begin','end'):init.append(dict(id=i+1,phase=phase,route='fp8_v_m16',instance=str(i),rows=1024,columns=5120,policy=16,context_present=phase=='end',factory_called=phase=='end',result=0 if phase=='end' else -1,exception=False))
        return factory,init
    def raw(self,factory,init):return '\n'.join(['GARNET_VM16_FACTORY '+json.dumps(row) for row in factory]+['GARNET_MODELOPT_INIT '+json.dumps(row) for row in init])
    def test_paired_API_configuration_live_opaque_and_both_projection_initializations(self):
        f,i=self.fixture();q=route_trace(self.raw(f,i));self.assertEqual(q['created_contexts'],1);self.assertEqual(q['initializations_by_output_rows'],{'1024':2});self.assertTrue(q['original_4MiB_workspace_preserved'])
    def test_all_seven_attributes_status_checks_workspace_and_opaque_required(self):
        for mutation in ('configuration','bytes','API','workspace','opaque','sequence'):
            f,i=self.fixture()
            if mutation=='configuration':next(r for r in f if r['event']=='configuration' and r['name']=='TILE_ID')['a']=15
            elif mutation=='bytes':next(r for r in f if r['event']=='configuration')['b']=3
            elif mutation=='API':next(r for r in f if r['event']=='api')['a']=7
            elif mutation=='workspace':next(r for r in f if r['event']=='check' and r['name']=='M16')['b']=(4<<20)+1
            elif mutation=='opaque':next(r for r in f if r['event']=='selection')['opaque']='cd'*64
            else:f[-1]['seq']+=1
            with self.assertRaises(AssertionError):route_trace(self.raw(f,i))
    def test_both_routes_policy_geometry_success_and_created_context_required(self):
        for mutation in ('missing','policy','geometry','exception','no_factory'):
            f,i=self.fixture()
            if mutation=='missing':i=i[:2]
            elif mutation=='policy':i[-1]['policy']=4
            elif mutation=='geometry':i[-1]['rows']=6144
            elif mutation=='exception':i[-1]['exception']=True
            else:f=[]
            with self.assertRaises(AssertionError):route_trace(self.raw(f,i))
    def test_reference_and_off_cannot_silently_select_V_route(self):
        f,i=self.fixture();self.assertEqual(assert_disabled('unrelated component output\n')['status'],'V_M16_route_absent_in_reference_or_off')
        for a,b in [(f,[]),([],i),(f,i)]:
            with self.assertRaises(AssertionError):assert_disabled(self.raw(a,b))
if __name__=='__main__':unittest.main()
