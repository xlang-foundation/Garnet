import json,unittest
from .kv128_projection_replay import factory_replay,CONFIG,PREFIX,APIS
class Projection(unittest.TestCase):
    def fixture(self):
        rows=[]
        def add(event,name,a=0,b=0,c=0,opaque=''):
            rows.append(dict(id=1,seq=len(rows)+1,event=event,name=name,a=a,b=b,c=c,opaque=opaque))
        add('begin','factory')
        for name in APIS:add('api',name,130400 if name=='cublasLtGetVersion' else 0)
        for n,a,b,c in CONFIG:add('configuration',n,a,b,c)
        add('selection','documented_M32_control',opaque='00'*64);add('check','M32');add('check','M128');add('accept','created');add('end','created')
        return rows
    def raw(self,rows):return b'\n'.join(PREFIX+json.dumps(r).encode() for r in rows)
    def test_successful_complete_checks(self):self.assertTrue(factory_replay(self.raw(self.fixture()),True)['all_successful'])
    def test_configuration_width_value_and_actual_check_faults(self):
        for name,field in [('ID','a'),('CTA_SWIZZLING','b'),('M128','a'),('M32','b')]:
            rows=self.fixture();row=next(r for r in rows if r['name']==name);row[field]=9999999
            with self.assertRaises(AssertionError):factory_replay(self.raw(rows),True)
    def test_incomplete_or_rejected_trace_never_admits(self):
        for mutate in ('missing','reject'):
            rows=self.fixture()
            if mutate=='missing':rows.pop()
            else:rows[-1]['name']='empty'
            with self.assertRaises(AssertionError):factory_replay(self.raw(rows),True)
if __name__=='__main__':unittest.main()
