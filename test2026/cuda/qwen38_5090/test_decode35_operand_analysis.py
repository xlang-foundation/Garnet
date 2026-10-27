import copy,hashlib,json,struct,unittest
from unittest.mock import patch
from . import decode35_operand_analysis as a
from .test_step45_layer_boundary_analysis import fixture as layer_fixture
from . import test_b4_full4k_selection52 as selection_fixtures

class Contract(unittest.TestCase):
    def test_exact_bf16_round_nearest_even_and_nonfinite_rejection(self):
        bits=[0,0x80000000,0x3f808000,0x3f818000,0xbf808000,0x3fa80000]
        self.assertEqual(struct.unpack('<6H',a.bf16_rn(struct.pack('<6I',*bits))),(0,0x8000,0x3f80,0x3f82,0xbf80,0x3fa8))
        for value in (0x7f800000,0xff800000,0x7fc00000,0x7f7fffff):
            with self.assertRaises(AssertionError):a.bf16_rn(struct.pack('<I',value))

    def test_native_trace_requires_one_exact_on_and_none_off(self):
        row={'batch':4,'tokens':4,'layer':35,'stages':7,'row_words':106496};raw=('GARNET_DECODE35_STAGE_WITNESS '+json.dumps(row)+'\n').encode()
        self.assertEqual(a.witness_trace(raw,4,True)['events'],1);self.assertEqual(a.witness_trace(b'',1,False)['events'],0)
        for data,batch,on in ((raw,4,False),(b'',4,True),(raw+raw,4,True),(raw,1,True)):
            with self.assertRaises(AssertionError):a.witness_trace(data,batch,on)

    def test_new_wide_width_and_all64_row_geometry(self):
        for batch in (1,4):
            r,f=layer_fixture(batch);r['observer']['wide_width']+=106496
            self.assertEqual(len(a.checked_layers(r,f.__getitem__,'selected',True)[45]),64)
            with self.assertRaises(AssertionError):a.checked_layers(r,f.__getitem__,'selected',False)
            r['observer']['wide_width']+=1
            with self.assertRaises(AssertionError):a.checked_layers(r,f.__getitem__,'selected',True)

    def test_DLL_transition_is_explicit_off_only_and_keeps_primary_exact(self):
        _,key,_,r,files,*_=selection_fixtures.Selection52().fixture();r['mode']='candidate';r['library_sha256']='a'*64;r['operand_witness']=False
        r['rows'][0].update(prompt_cache={},key35_cache52={},key35_coordinate_trajectory={})
        r['pool']={k:0 for k in ('embedding_mode','head_precision','compressed_projection_pool_entries','compressed_nvfp4_entries','compressed_fp8_entries')}
        old=copy.deepcopy(r);old['library_sha256']='b'*64
        with patch.object(a,'checked_cache',return_value=b'cache'),patch.object(a,'checked_trajectory'),patch.object(a,'checked_layers',return_value={}):
            q=a.primary_control(r,old,files.__getitem__,files.__getitem__,key,'a'*64,'b'*64,False,True);self.assertTrue(q['declared_off_DLL_transition'])
            with self.assertRaises(AssertionError):a.primary_control(r,old,files.__getitem__,files.__getitem__,key,'a'*64,'b'*64,True,True)
            with self.assertRaises(AssertionError):a.primary_control(r,old,files.__getitem__,files.__getitem__,key,'a'*64,'c'*64,False,True)
            r['rows'][0]['post_step_states'][0]['capture']['logical']['components'][127]['row_sha256'][0]='f'*64
            with self.assertRaises(AssertionError):a.primary_control(r,old,files.__getitem__,files.__getitem__,key,'a'*64,'b'*64,False,True)

    def test_any_control_failure_prevents_all_operand_interpretation(self):
        reports={p:{} for p in a.PHASES};old={r:{} for r in ('reference','candidate')};reads={p:None for p in a.PHASES}
        with patch.object(a,'primary_control',side_effect=AssertionError('primary drift')),patch.object(a,'checked_operands') as operand:
            with self.assertRaisesRegex(AssertionError,'primary drift'):a.analyze({},'key',reports,old,reads,old,'a'*64,{'reference':'b'*64,'candidate':'c'*64})
            operand.assert_not_called()

    def operand_fixture(self):
        r,f=layer_fixture(1);r['operand_witness']=True;r['observer']['wide_width']+=106496
        cursor=r['observer']['primary_width']+64*20480;offsets=[]
        for name,width in a.OPERANDS:offsets.append([name,cursor,width]);cursor+=width
        r['internal_observer']={'names':[n for n,_ in a.OPERANDS],'widths':[w for _,w in a.OPERANDS],'offsets':offsets,'row_words':106496}
        row=r['rows'][0];row['final_state']={'logical':{'components':[{} for _ in range(128)]}};row['key35_cache52']={};row['operand_observations']=[]
        for step in (44,45):
            e={'step':step,'program_runs':step,'consumed_tokens':[100+step-1],'operands':{}}
            for name,width in a.OPERANDS:
                filename=f'operand-{step}-{name}.f32';raw=bytes(width*4);f[filename]=raw;e['operands'][name]={'file':filename,'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}
            row['operand_observations'].append(e)
        return r,f

    def test_all_operands_link_masked_input_suffix_and_actual_cache(self):
        r,f=self.operand_fixture()
        with patch.object(a,'checked_cache',return_value=bytes(8388608)):
            self.assertEqual(set(a.checked_operands(r,f.__getitem__,'selected')),{44,45})
        for mutation in ('masked','suffix','cache','hash','offset'):
            c,g=self.operand_fixture()
            if mutation in ('masked','suffix','cache'):
                name='masked_input' if mutation=='masked' else 'K_post_RoPE';entry=c['rows'][0]['operand_observations'][1]['operands'][name];raw=bytearray(g[entry['file']]);word=0 if mutation in ('masked','cache') else 211;raw[word*4:word*4+4]=struct.pack('<f',1);g[entry['file']]=bytes(raw);entry['sha256']=hashlib.sha256(raw).hexdigest()
            elif mutation=='hash':c['rows'][0]['operand_observations'][0]['operands']['projected_K']['sha256']='0'*64
            else:c['internal_observer']['offsets'][0][1]+=1
            with patch.object(a,'checked_cache',return_value=bytes(8388608)):
                with self.assertRaises(AssertionError):a.checked_operands(c,g.__getitem__,'selected')

    def test_operand_scope_row_or_consumed_ID_tampering_rejected(self):
        for mutation in ('names','step','ID'):
            r,f=self.operand_fixture()
            if mutation=='names':r['internal_observer']['names'][0]='wrong'
            elif mutation=='step':r['rows'][0]['operand_observations'][0]['step']=43
            else:r['rows'][0]['operand_observations'][0]['consumed_tokens']=[999]
            with patch.object(a,'checked_cache',return_value=bytes(8388608)):
                with self.assertRaises(AssertionError):a.checked_operands(r,f.__getitem__,'selected')

if __name__=='__main__':unittest.main()
