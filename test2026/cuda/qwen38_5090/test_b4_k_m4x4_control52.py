from pathlib import Path
import copy,hashlib,struct,unittest
from . import test_b4_full4k_selection52 as fixtures
from .test_b4_k35_cache52 import layout
from .b4_k35_cache52 import CACHE_BYTES,key_layout
from .bf16_float_reference_state import expand_bf16_carrier
from .b4_k_m4x4_control52 import control,intervention
sha=lambda b:hashlib.sha256(b).hexdigest()

def add_cache(report,raw,position,files,label):
    row=report['rows'][0];digest=sha(expand_bf16_carrier(raw))
    for capture in (row['final_state'],row['post_step_states'][-1]['capture']):capture['logical']['components'][70]['row_sha256'][position]=digest
    name=label+'.bf16carrier';files[name]=raw
    row['key35_cache52']=dict(file=name,bytes=CACHE_BYTES,sha256=sha(raw),repeat_sha256=[sha(raw)]*2,logical_sha256=digest,physical_row=position,geometry=key_layout(layout()),storage='little_endian_two_BF16_per_F32_carrier',step=52)

def fixture():
    s,key,r,c,f,fullr,fullc,er,ec=fixtures.Selection52().fixture()
    for report,mode in ((r,'reference'),(c,'candidate')):report.update(mode=mode,library_sha256='a'*64)
    add_cache(r,bytes(CACHE_BYTES),0,f,'reference');add_cache(c,bytes(CACHE_BYTES),2,f,'candidate')
    c['intervention_policy']='on'
    return s,key,r,c,f,fullr,ec

class Control52(unittest.TestCase):
    def test_flagoff_all_rows_and_last_component_ID_cache_and_binary_guards(self):
        s,key,r,c,f,fr,ec=fixture();c['intervention_policy']='off';old=copy.deepcopy(c);c['library_sha256']='b'*64
        self.assertEqual(control(c,old,lambda n:f[n],lambda n:f[n],'b'*64,'a'*64)['complete_captures'],5)
        for mutation in ('ID','component','binary','final_alias','cache'):
            actual=copy.deepcopy(c);files=dict(f)
            if mutation=='ID':actual['rows'][0]['generated_ids'][0][52]+=1
            elif mutation=='component':actual['rows'][0]['post_step_states'][0]['capture']['logical']['components'][127]['row_sha256'][3]='f'*64
            elif mutation=='binary':actual['library_sha256']='c'*64
            elif mutation=='final_alias':actual['rows'][0]['post_step_states'][1]['capture']['logical']['components'][127]['row_sha256'][3]='f'*64
            else:name=actual['rows'][0]['key35_cache52']['file'];files[name]=f[name][:-2]+b'\x80\x3f'
            with self.assertRaises(AssertionError):control(actual,old,lambda n:files[n],lambda n:f[n],'b'*64,'a'*64)
    def test_exact_selected_state_cache_and_all_four_IDs_pass_only_narrow_intervention(self):
        s,key,r,c,f,fr,ec=fixture();c['rows'][0]['generated_ids'][2][52]=r['rows'][0]['generated_ids'][0][52]
        q=intervention(s,key,r,c,fr,ec,lambda n:f[n],lambda n:f[n],lambda n:f[n],lambda n:f[n]);self.assertTrue(q['selected_step52_intervention_passed']);self.assertEqual(q['all_four_53_IDs_exact'],[True]*4);self.assertFalse(q['full128_output_4K_MTP_quality_HTTP_or_performance_qualified'])
    def test_prediction_component_or_one_cache_value_prevents_promotion(self):
        for mutation in ('ID','component','cache'):
            s,key,r,c,f,fr,ec=fixture();c['rows'][0]['generated_ids'][2][52]=r['rows'][0]['generated_ids'][0][52]
            if mutation=='ID':c['rows'][0]['generated_ids'][2][52]+=1
            elif mutation=='component':
                for capture in (c['rows'][0]['final_state'],c['rows'][0]['post_step_states'][-1]['capture']):capture['logical']['components'][127]['row_sha256'][2]='f'*64
            else:
                raw=bytearray(CACHE_BYTES);raw[-2:]=struct.pack('<H',0x3f80);add_cache(c,bytes(raw),2,f,'different-candidate')
            q=intervention(s,key,r,c,fr,ec,lambda n:f[n],lambda n:f[n],lambda n:f[n],lambda n:f[n]);self.assertFalse(q['selected_step52_intervention_passed']);self.assertFalse(q['full128_output_4K_MTP_quality_HTTP_or_performance_qualified'])
if __name__=='__main__':unittest.main()
