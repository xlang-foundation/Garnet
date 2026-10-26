import copy,hashlib,json,unittest
from pathlib import Path
from .z128_reconstructed_layer2_replay import replay,require_admission
from .z128_reconstructed_layer2_layout import check_graph,INPUT_SOURCE
from .layer2_ordinary_replay import array
from .test_gdn_chunk_stage_witness import fixture,entry
def sample():
    report,files=fixture();report=copy.deepcopy(report);report.update(status='layer2_ordinary_collected_needs_replay',input_source=INPUT_SOURCE,frontend_rejections=[],common_input=report['prompt_input'])
    report['cases']=[r for r in report['cases'] if r['layer']==1 and r['diagnostic'] is False];common=array(report['common_input'],files.__getitem__);report['input_fixture_sha256']=hashlib.sha256(common).hexdigest()
    for r in report['cases']:r.update(layer=2,input_lineage=INPUT_SOURCE);r['graph'].update(native_fp8_z128_canonical_projections=0,native_fp8_z128_reconstructed_projections=int(r['tokens']==128))
    return report,files,common
class Controls(unittest.TestCase):
    def test_actual_native_trace_envelope_rejects_wrong_plugin_family(self):
        from .z128_reconstructed_layer2_trace import replay as trace
        p=dict(id=1,phase='begin',route='fp8_z128_reconstructed',instance='cpu',rows=6144,columns=5120,policy=0x52313238,context_present=False,factory_called=False,result=-1,exception=False)
        f=dict(id=1,seq=1,event='begin',name='factory',a=0,b=0,c=0,opaque='')
        rows=[('GARNET_MODELOPT_INIT ',p),('GARNET_Z128_RECONSTRUCT ',f),('GARNET_Z128_RECONSTRUCT ',dict(f,seq=2,event='api',name='cudaGetDevice',a=7)),('GARNET_Z128_RECONSTRUCT ',dict(f,seq=3,event='reject',name='device_or_version')),('GARNET_Z128_RECONSTRUCT ',dict(f,seq=4,event='end',name='empty')),('GARNET_MODELOPT_INIT ',dict(p,phase='end',factory_called=True,result=1))]
        encode=lambda r:'\n'.join(a+json.dumps(b) for a,b in r)
        self.assertEqual(trace(encode(rows),True)['factory']['calls'][0]['outcome'],'empty')
        bad=copy.deepcopy(rows);bad[0][1]['route']='fp8_z128'
        with self.assertRaises(AssertionError):trace(encode(bad),True)
        with self.assertRaises(AssertionError):trace(encode(rows),False)
    def test_exact_collector_derivative_preserves_every_array_and_math_operation(self):
        m=Path(__file__).resolve().parent;p=json.loads((m/'evidence/z128-reconstructed-layer2-derivation.json').read_bytes())
        owner=json.loads((m/'evidence/z128-reconstructed-layer2-owner-derivation.json').read_bytes())
        for row in p['derivations']+[owner]:
            raw=(m/row['source']).read_bytes();self.assertEqual(hashlib.sha256(raw).hexdigest(),row['source_sha256'])
            for c in row['changes']:
                a=c['before'].encode();self.assertEqual(raw.count(a),c['count']);raw=raw.replace(a,c['after'].encode())
            self.assertEqual(raw,(m/row['target']).read_bytes());self.assertEqual(hashlib.sha256(raw).hexdigest(),row['target_sha256'])
    def test_same_arrays_pass_without_modifying_actual_metadata(self):
        r,f,c=sample();old=copy.deepcopy(r);q=replay(r,f.__getitem__,c);self.assertTrue(require_admission(q));self.assertEqual(old,r)
    def test_every_correctly_rehashed_hidden_history_and_state_difference_is_rejected(self):
        for offset in (0,128*5120*4,(128*5120+40960)*4):
            r,f,c=sample();row=r['cases'][1];raw=bytearray(array(row['repeats'][0][0]['arrays']['primary'],f.__getitem__));raw[offset]^=1;e,data=entry(bytes(raw))
            for part in e['parts']:old=part['file'];part['file']='fault-'+old;f[part['file']]=data[old]
            for seq in row['repeats']:seq[0]['arrays']['primary']=e
            q=replay(r,f.__getitem__,c);self.assertFalse(q['layer2_cross_profile_admitted'])
            with self.assertRaises(AssertionError):require_admission(q)
    def test_legacy_or_missing_or_bool_new_counters_fail(self):
        r,_,_=sample();base=r['cases'][1]['graph']
        for key,value in [('native_fp8_z128_canonical_projections',1),('native_fp8_z128_reconstructed_projections',0),('native_fp8_z128_reconstructed_projections',True)]:
            with self.assertRaises(AssertionError):check_graph(dict(base,**{key:value}),128,False)
        bad=dict(base);del bad['native_fp8_z128_reconstructed_projections']
        with self.assertRaises(KeyError):check_graph(bad,128,False)
