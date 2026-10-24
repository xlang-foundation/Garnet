"""CPU contract tests; no Garnet/CUDA imports or timing claims."""
import ast,copy,json,unittest
from pathlib import Path
from xModel.qwen35.vl_0_8b.kv_delta_layout import (
    build_kv_delta_layout,choose_kv_overlay_strategy,validate_kv_delta_profile)

ROOT=Path(__file__).resolve().parents[3]
CONFIG=json.loads((Path(__file__).parent/'evidence/nvfp4-config.json').read_text())['text_config']
source=ROOT/'xModel/qwen35/vl_0_8b/cuda_packed_program.py'
function=next(n for n in ast.parse(source.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='decoder_state_layout')
namespace={}
exec(compile(ast.Module(body=[function],type_ignores=[]),str(source),'exec'),namespace)

def baseline():return namespace['decoder_state_layout'](CONFIG,4,4096)
def flags():return dict(external_compressed_weights=True,retain_linear_inputs=True,deferred_state_commit=True,shared_output_storage=True,joined_mlp_projections=True,host_embedding_lookup=True)

class KVDeltaLayoutTests(unittest.TestCase):
    def test_actual_canonical_offsets_and_logical_output_sizes(self):
        canonical,width=baseline();saved=copy.deepcopy(canonical)
        compact=build_kv_delta_layout(canonical)
        self.assertEqual(canonical,saved)
        self.assertEqual(compact['canonical_state_width'],width)
        self.assertEqual(width*4*4,2782920704)
        self.assertEqual(compact['linear_replay_width']*4*4,635437056)
        self.assertEqual(compact['compact_state_width']*4*4,637534208)
        self.assertEqual(compact['output_width']*4*4,688488448)
        self.assertEqual(compact['payload_width']*4*4,50626560)
        for old,new in zip(canonical,compact['layers']):
            self.assertEqual(new['canonical_offset'],old['offset'])
            if old['kind']=='full_attention':
                self.assertEqual(new['cache_batch_stride'],width)
                self.assertEqual(new['value_offset'],new['key_offset']+new['cache_row_width'])
                self.assertLessEqual(new['value_offset']+new['cache_row_width'],width)
                self.assertNotIn('replay_offset',new)
                self.assertEqual(new['output_state1_shape'],[4,4,4,256])

    def test_output_and_replay_regions_cover_the_declared_buffers(self):
        layout=build_kv_delta_layout(baseline()[0]);out=layout['hidden_width'];replay=0
        for layer in layout['layers']:
            self.assertEqual(layer['output_offset'],out)
            out+=layer['output_state1_width']+layer['output_state2_width']
            if layer['kind']=='linear_attention':
                self.assertEqual(layer['replay_offset'],replay)
                replay+=layer['canonical_state1_width']+layer['canonical_state2_width']
        for layer in layout['layers']:
            if layer['kind']=='linear_attention':
                self.assertEqual(layer['payload_offset'],out);out+=layer['payload_width']
        self.assertEqual(out,layout['output_width']);self.assertEqual(replay,layout['linear_replay_width'])

    def test_bad_offsets_shapes_widths_and_unqualified_geometry_rejected(self):
        canonical,_=baseline()
        for field,value in [('offset',1),('offset',False),('index',True),('state1_width',1),('state1_shape',[4,10240,True]),('kind','full_attention')]:
            bad=copy.deepcopy(canonical);bad[0][field]=value
            with self.subTest(field=field,value=value),self.assertRaises(ValueError):build_kv_delta_layout(bad)
        for args in ((1,4096,4,5120),(4,2048,4,5120),(4,4096,16,5120),(True,4096,4,5120),(4,4096.0,4,5120)):
            with self.subTest(args=args),self.assertRaises(ValueError):build_kv_delta_layout(canonical,*args)
        with self.assertRaises(ValueError):build_kv_delta_layout(canonical[:-1])

    def test_explicit_profile_and_disabled_path(self):
        validate_kv_delta_profile(True,4,4096,4,4,**flags())
        validate_kv_delta_profile(False,8,8192,128,16,**{k:False for k in flags()})
        for key in flags():
            bad=flags();bad[key]=1
            with self.subTest(key=key),self.assertRaises(ValueError):validate_kv_delta_profile(True,4,4096,4,4,**bad)
        for args in ((1,4,4096,4,4),(True,4,4096,128,4),(True,4,4096,4,16),(True,True,4096,4,4)):
            with self.subTest(args=args),self.assertRaises(ValueError):validate_kv_delta_profile(*args,**flags())
        with self.assertRaises(ValueError):validate_kv_delta_profile(True,4,4096,4,4,**flags(),embedded_projection_weights=True)

    def test_dispatch_boundary_mixed_rows_and_inactive_capacity_slots(self):
        self.assertEqual(choose_kv_overlay_strategy([508,509,510,511]*4,[1]*16),0)
        self.assertEqual(choose_kv_overlay_strategy([509,510,511,512]*4,[1]*16),1)
        slots=[0,1,2,3]*3+[4095,4096,4096,4096]
        self.assertEqual(choose_kv_overlay_strategy(slots,[1]*12+[1,0,0,0]),1)
        self.assertEqual(choose_kv_overlay_strategy([4096]*16,[0]*16),0)
        for slots,active in [([0]*16,[1]*16),([4096]*16,[1]*16),([False]*16,[0]*16),([0]*16,[True]*16),([0.0]*16,[0]*16),([0]*15,[0]*15)]:
            with self.subTest(slots=slots,active=active),self.assertRaises(ValueError):choose_kv_overlay_strategy(slots,active)

if __name__=='__main__':unittest.main()
