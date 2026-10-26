"""Fault tests for a controlled ordinary observation; no broad model/performance admission."""
from pathlib import Path
import ast,copy,hashlib,json,struct,unittest
from .layer2_ordinary_layout import validate_profile
from .layer2_ordinary_replay import extract_common_input,replay,array
sha=lambda raw:hashlib.sha256(raw).hexdigest()
class Controls(unittest.TestCase):
    def test_profile_rejects_diagnostic_and_geometry_drift(self):
        for tokens in (32,128):self.assertEqual(validate_profile(2,tokens,False)['tokens'],tokens)
        for values in ((True,32,False),(0,32,False),(1,32,False),(3,32,False),(2,16,False),(2,True,False),(2,32,0),(2,32,True),(2,32,None)):
            with self.assertRaises(ValueError):validate_profile(*values)
    def test_common_input_extraction_requires_real_complete_repeat_exact(self):
        files={};chunks=[]
        for i in range(4):
            primary=struct.pack('<f',float(i+1))*(32*5120)+bytes((40960+786432)*4);name=str(i)+'.f32';files[name]=primary;entry={'words':len(primary)//4,'sha256':sha(primary),'parts':[{'file':name,'offset':0,'words':len(primary)//4,'sha256':sha(primary)}]};chunks.append({'start':i*32,'repeat':0,'arrays':{'primary':entry}})
        repeat=copy.deepcopy(chunks)
        for row in repeat:row['repeat']=1
        report={'status':'gdn_chunk_stage_collected_needs_replay','model_or_performance_qualified':False,'cases':[{'layer':1,'tokens':32,'diagnostic':False,'input_lineage':'retained_T32_layer0_ordinary_output_common_input','repeats':[chunks,repeat]}]}
        self.assertEqual(len(extract_common_input(report,files.__getitem__)),2621440)
        for fault in ('missing','start','repeat','hash','different_repeat'):
            bad=copy.deepcopy(report);row=bad['cases'][0]
            if fault=='missing':row['repeats'][0].pop()
            elif fault=='start':row['repeats'][0][0]['start']=1
            elif fault=='repeat':row['repeats'][1][0]['repeat']=0
            elif fault=='hash':row['repeats'][0][0]['arrays']['primary']['sha256']='f'*64
            else:row['repeats'][1][0]['arrays']['primary']=copy.deepcopy(row['repeats'][1][1]['arrays']['primary'])
            with self.assertRaises(AssertionError):extract_common_input(bad,files.__getitem__)
    def test_exact_method_derivations_keep_math_and_old_guards(self):
        m=Path(__file__).resolve().parent;record=json.loads((m/'evidence/layer2-ordinary-witness-derivation.json').read_bytes());self.assertFalse(record['native_or_model_math_changed'])
        for row in record['derivations']:
            raw=(m/row['base']).read_bytes();self.assertEqual(sha(raw),row['base_sha256'])
            for change in row['changes']:
                before=change['before'].encode();self.assertEqual(raw.count(before),change['count']);raw=raw.replace(before,change['after'].encode())
            self.assertEqual(raw,(m/row['target']).read_bytes());self.assertEqual(sha(raw),row['target_sha256'])
            if row['target'].endswith('.py'):ast.parse(raw)
    def test_full_replay_reports_arithmetic_difference_and_rejects_mutated_input(self):
        from .test_gdn_chunk_stage_witness import fixture,entry
        from .layer2_ordinary_layout import INPUT_SOURCE
        parent,files=fixture();report=copy.deepcopy(parent);report.update(status='layer2_ordinary_collected_needs_replay',input_source=INPUT_SOURCE,frontend_rejections=[],common_input=parent['prompt_input'])
        report['cases']=[row for row in report['cases'] if row['layer']==1 and row['diagnostic'] is False]
        common=array(report['common_input'],files.__getitem__);report['input_fixture_sha256']=sha(common)
        for row in report['cases']:row.update(layer=2,input_lineage=INPUT_SOURCE);row['graph']['native_fp8_z128_canonical_projections']=int(row['tokens']==128)
        self.assertTrue(replay(report,files.__getitem__,common)['hidden_bytes_exact'])
        target=report['cases'][1]
        changed=bytearray(array(target['repeats'][0][0]['arrays']['primary'],files.__getitem__));changed[0]^=1;new,data=entry(bytes(changed))
        for part in new['parts']:
            old=part['file'];part['file']='layer2fault-'+old;files[part['file']]=data[old]
        for sequence in target['repeats']:sequence[0]['arrays']['primary']=new
        result=replay(report,files.__getitem__,common);self.assertFalse(result['hidden_bytes_exact']);self.assertEqual(result['different_hidden_f32_words'],1);self.assertTrue(result['final_state_exact']);self.assertFalse(result['model_or_quality_or_performance_qualified'])
        target['repeats'][0][0]['input_hidden_after']=new
        with self.assertRaises(AssertionError):replay(report,files.__getitem__,common)
if __name__=='__main__':unittest.main()
