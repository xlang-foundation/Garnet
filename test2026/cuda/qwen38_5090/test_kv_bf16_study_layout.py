"""CPU byte-layout boundaries, separate from CUDA math or VRAM qualification."""
import importlib.util,unittest
from pathlib import Path

path=Path(__file__).resolve().parents[3]/'xModel/qwen35/vl_0_8b/kv_bf16_study_layout.py'
spec=importlib.util.spec_from_file_location('bf16_study_layout',path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
build=module.build_bf16_kv_study_layout;offset=module.bf16_kv_byte_offset

class LayoutTests(unittest.TestCase):
    def test_logical_sizes_and_shared_prefill_output_bound(self):
        decode=build();prefill=build(chunk=32,retain_linear_inputs=False)
        self.assertEqual(decode['canonical_state_width'],106823680)
        self.assertEqual(decode['canonical_state_width']*8*4,3418357760)
        self.assertEqual((173932544-decode['canonical_state_width'])*8*4,2147483648)
        self.assertEqual(decode['output_width'],43030528)
        self.assertEqual(prefill['output_width'],40927232)
        self.assertLess(prefill['output_width'],decode['output_width'])
        self.assertEqual(decode['linear_replay_width']*8*4,1270874112)
        self.assertFalse(decode['runtime_qualified'])
    def test_every_layer_is_contiguous_and_gdn_precision_stays_fp32(self):
        for batch in (4,8):
            layout=build(batch=batch);cursor=0;linear=full=0
            for index,layer in enumerate(layout['layers']):
                self.assertEqual(layer['index'],index);self.assertEqual(layer['offset'],cursor)
                cursor+=layer['state1_width']+layer['state2_width']
                if index%4==3:
                    full+=1;self.assertEqual(layer['state_precision'],'packed_bf16')
                    self.assertEqual(layer['logical_kv_shape'],[batch,4,4096,256])
                    self.assertEqual(layer['state1_width']*4,4*4096*256*2)
                else:
                    linear+=1;self.assertEqual(layer['state_precision'],'fp32')
                    self.assertEqual(layer['state1_width']+layer['state2_width'],827392)
            self.assertEqual((linear,full,cursor),(48,16,106823680))
    def test_first_last_and_cross_head_cache_byte_boundaries(self):
        layout=build();first=layout['layers'][3];last=layout['layers'][63]
        self.assertEqual(offset(layout,3,'key',0,0,0,0),3*827392*4)
        self.assertEqual(offset(layout,3,'key',0,0,0,1)-offset(layout,3,'key',0,0,0,0),2)
        self.assertEqual(offset(layout,3,'key',0,1,0,0)-offset(layout,3,'key',0,0,0,0),4096*256*2)
        self.assertEqual(offset(layout,3,'key',0,3,4095,255)+2,offset(layout,3,'value',0,0,0,0))
        self.assertEqual(offset(layout,63,'value',7,3,4095,255)+2,3418357760)
        self.assertEqual(offset(layout,3,'value',0,3,4095,255)+2,4*(first['offset']+first['state1_width']+first['state2_width']))
        self.assertEqual(last['offset']+last['state1_width']+last['state2_width'],106823680)
    def test_target_payload_and_replay_regions_do_not_overlap(self):
        for chunk,retain in ((4,True),(32,False)):
            layout=build(chunk=chunk,retain_linear_inputs=retain);target=chunk*5120;replay=0;payload=layout['target_state_end']
            for layer in layout['layers']:
                self.assertEqual(layer['target_offset'],target);target+=layer['output_state1_width']+layer['output_state2_width']
                if layer['kind']=='linear_attention':
                    self.assertEqual(layer['replay_offset'],replay);replay+=827392
                    if retain:
                        self.assertEqual(layer['payload_offset'],payload);payload+=65920
                else:self.assertIsNone(layer['replay_offset'])
            self.assertEqual(target,layout['target_state_end']);self.assertEqual(payload,layout['output_width'])
    def test_unqualified_profiles_and_noninteger_indices_reject(self):
        for args in ({'batch':True},{'batch':2},{'capacity':2048},{'chunk':8},{'chunk':32},{'retain_linear_inputs':1}):
            with self.subTest(args=args),self.assertRaises(ValueError):build(**args)
        layout=build()
        for args in ((3,'key',True,0,0,0),(0,'key',0,0,0,0),(63,'value',8,0,0,0),(3,'other',0,0,0,0),(3,'key',0,4,0,0),(3,'key',0,0,4096,0),(3,'key',0,0,0,256)):
            with self.subTest(args=args),self.assertRaises(ValueError):offset(layout,*args)
    def test_old_contract_cannot_be_used_as_packed_bf16(self):
        layout=build();layout['contract']='garnet_qwen38_fp32_kv_delta_v1'
        with self.assertRaises(ValueError):offset(layout,3,'key',0,0,0,0)

if __name__=='__main__':unittest.main()
