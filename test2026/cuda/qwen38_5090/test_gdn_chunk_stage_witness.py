"""PureCPU bounded full-word/provenance and native graph metadata faults."""
from pathlib import Path
import copy,hashlib,json,struct,unittest
import numpy as np
from .gdn_chunk_stage_layout import layout,pieces,READER_WORDS,check_graph
from .gdn_chunk_stage_replay import array,replay
sha=lambda raw:hashlib.sha256(raw).hexdigest()

def entry(raw):
    files={};parts=[]
    for offset,words in pieces(len(raw)//4):
        value=raw[offset*4:(offset+words)*4];name='part-'+str(offset)+'.f32';files[name]=value;parts.append({'file':name,'offset':offset,'words':words,'sha256':sha(value)})
    return {'words':len(raw)//4,'sha256':sha(raw),'parts':parts},files

def fixture():
    # Synthetic byte controls only; never reported as hardware/model observations.
    files={};cache={};counter=0
    def store(raw):
        nonlocal counter
        result=[]
        for offset,words in pieces(len(raw)//4):
            value=raw[offset*4:(offset+words)*4];name='synthetic-'+str(counter)+'.f32';counter+=1;files[name]=value;result.append({'file':name,'offset':offset,'words':words,'sha256':sha(value)})
        return {'words':len(raw)//4,'sha256':sha(raw),'parts':result}
    def zeros(words):
        if words not in cache:cache[words]=bytes(words*4)
        return cache[words]
    prompt=np.repeat(np.arange(128,dtype='<f4'),5120).tobytes();history=zeros(40960);state=zeros(786432)
    names=['flag2','flag_negative','flag_bool','flag_float','legacy_conflict','batch2','tokens16','layer2','missing_active','bf16_hidden']
    report={'status':'gdn_chunk_stage_collected_needs_replay','model_or_performance_qualified':False,'prompt_tokens':3968,'input_ids':list(range(128)),'input_source':'first128_actual_text_prompt_embeddings','frontend_rejections':[{'name':name,'rejected':True,'error':'synthetic rejection'} for name in names],'prompt_input':store(prompt),'cases':[]}
    for layer in (0,1):
        for t in (32,128):
            for diagnostic in (False,True):
                spec=layout(t,diagnostic);graph={'checkpoint_weight_precision':'modelopt_mixed','native_nvfp4_projection_weights':2,'native_fp8_projection_weights':3,'external_compressed_weight_inputs':5,'native_int4_g128_projection_weights':0,'native_fp8_m128_canonical_projections':int(t==128),'device_output_bytes':spec['total_words']*4,'graph_capture_failures':0,'graph_captures':1,'graph_replays':2}
                row={'layer':layer,'tokens':t,'diagnostic':diagnostic,'layout':spec,'graph':graph,'input_lineage':'prompt_embeddings' if layer==0 else 'retained_T32_layer0_ordinary_output_common_input','active':store(struct.pack('<f',1)*t),'repeats':[]};report['cases'].append(row)
                for repeat in (0,1):
                    sequence=[];row['repeats'].append(sequence)
                    for start in range(0,128,t):
                        hidden=prompt[start*5120*4:(start+t)*5120*4];primary=hidden+history+state
                        arrays={part['name']:store(primary if part['name']=='primary' else hidden if part['name'] in ('masked_input','output_hidden') else zeros(part['words'])) for part in spec['rows']}
                        sequence.append({'start':start,'repeat':repeat,'input_hidden':store(hidden),'input_history':store(history),'input_state':store(state),'input_hidden_after':store(hidden),'input_history_after':store(history),'input_state_after':store(state),'arrays':arrays})
    return report,files

class Controls(unittest.TestCase):
    def test_actual_oversized_primary_and_qkv_split_without_reader_expansion(self):
        p=layout(128,True);self.assertEqual(p['primary_words'],1482752);self.assertEqual(next(r['words'] for r in p['rows'] if r['name']=='qkv'),1310720)
        self.assertEqual(pieces(1482752),[(0,1048576),(1048576,434176)])
        for t in (32,128):
            for diag in (False,True):
                spec=layout(t,diag);self.assertEqual(spec['rows'][0]['offset'],0);self.assertEqual(sum(row['words'] for row in spec['rows']),spec['total_words'])
                self.assertTrue(all(0<words<=READER_WORDS for row in spec['rows'] for offset,words in pieces(row['words'])))

    def test_every_actual_piece_and_word_is_checked(self):
        raw=struct.pack('<f',1.25)*(READER_WORDS+1);e,f=entry(raw);self.assertEqual(array(e,f.__getitem__),raw)
        for fault in ('missing','gap','hash','path','size','boolean'):
            bad=copy.deepcopy(e)
            if fault=='missing':bad['parts'].pop()
            elif fault=='gap':bad['parts'][1]['offset']-=1
            elif fault=='hash':bad['parts'][1]['sha256']='0'*64
            elif fault=='path':bad['parts'][1]['file']='../outside.f32'
            elif fault=='size':bad['words']+=1
            else:bad['parts'][0]['offset']=False
            with self.subTest(fault=fault),self.assertRaises(AssertionError):array(bad,f.__getitem__)
        corrupted=dict(f);corrupted[e['parts'][1]['file']]=bytes(4)
        with self.assertRaises(AssertionError):array(e,corrupted.__getitem__)

    def test_nonfinite_values_fail_even_with_matching_piece_hashes(self):
        for x in (float('inf'),float('-inf'),float('nan')):
            e,f=entry(struct.pack('<f',x))
            with self.assertRaises(AssertionError):array(e,f.__getitem__)

    def test_native_geometry_count_output_and_capture_scope(self):
        base={'checkpoint_weight_precision':'modelopt_mixed','native_nvfp4_projection_weights':2,'native_fp8_projection_weights':3,'external_compressed_weight_inputs':5,'native_int4_g128_projection_weights':0,'native_fp8_m128_canonical_projections':1,'device_output_bytes':layout(128,True)['total_words']*4,'graph_capture_failures':0,'graph_captures':1,'graph_replays':2};check_graph(base,128,True)
        for key,value in (('native_fp8_m128_canonical_projections',0),('native_fp8_projection_weights',2),('native_nvfp4_projection_weights',True),('device_output_bytes',0),('graph_captures',0),('graph_capture_failures',1),('graph_replays',0)):
            with self.subTest(key=key),self.assertRaises(AssertionError):check_graph(dict(base,**{key:value}),128,True)

    def test_recorded_owner_is_exact_derivative_of_preserved_transaction(self):
        m=Path(__file__).resolve().parent;r=json.loads((m/'evidence/gdn-chunk-stage-witness-owner-derivation.json').read_bytes());source=(m/r['source']).read_bytes();self.assertEqual(sha(source),r['source_sha256']);text=source.decode()
        for step in r['replacements']:
            self.assertEqual(text.count(step['old']),step['count']);text=text.replace(step['old'],step['new'])
        self.assertEqual(text.encode(),(m/r['target']).read_bytes());self.assertEqual(sha(text.encode()),r['target_sha256'])

    def test_complete_synthetic_all_profile_replay(self):
        report,files=fixture();result=replay(report,files.__getitem__);self.assertEqual(result['profiles'],8);self.assertTrue(result['diagnostic_primary_exact_to_ordinary']);self.assertTrue(all(row['different_stages']==[] for row in result['stage_comparisons']))

    def test_changed_primary_with_fresh_hash_still_fails_witness(self):
        report,files=fixture();target=report['cases'][3]
        for sequence in target['repeats']:
            e=sequence[0]['arrays']['primary'];raw=bytearray(array(e,files.__getitem__));raw[128*5120*4]^=1
            changed,parts=entry(bytes(raw));renamed={}
            for part in changed['parts']:
                name='changed-'+str(len(files))+part['file'];files[name]=parts[part['file']];part['file']=name
            sequence[0]['arrays']['primary']=changed
        with self.assertRaises(AssertionError):replay(report,files.__getitem__)

if __name__=='__main__':unittest.main()
