import copy,json,unittest
from .test_b4_k_m4x4_control52 import fixture
from .test_b4_k35_coordinate_trajectory import fixture as trajectory
from .rowwise_k_norm_analysis import PHASES,norm_trace,control,analyze

def sample():
    s,key,r,c,files,fr,prefix=fixture()
    for report,batch,position in ((r,1,0),(c,4,2)):
        row=report['rows'][0];row['key35_coordinate_trajectory']=trajectory(batch,position,first=53)[0]
        report.update(primary_outputs_only=True,intervention_policy='off')
    c['rows'][0]['generated_ids'][2][52]=r['rows'][0]['generated_ids'][0][52]
    reports={'off-reference':r,'off-candidate':c,'on-candidate':copy.deepcopy(c)};reports['on-candidate']['intervention_policy']='on'
    return s,key,reports,{n:lambda name:files[name] for n in PHASES},fr,prefix,files

class Controls(unittest.TestCase):
    def test_all16_subgraphs_are_not_claimed_as_executed_kernels(self):
        raw='\n'.join('GARNET_ROWWISE_K_DECODE_NORM '+json.dumps(dict(batch=4,tokens=4,layer=i,rms_subgraphs=4)) for i in range(3,64,4))
        q=norm_trace(raw,True);self.assertEqual(q['full_attention_layers'],16);self.assertFalse(q['executed_kernel_count_claimed'])
        for bad in (raw+'\n'+raw.splitlines()[0],raw.replace('"tokens": 4','"tokens": 1'),raw.replace('"layer": 63','"layer": 62')):
            with self.assertRaises(AssertionError):norm_trace(bad,True)
        with self.assertRaises(AssertionError):norm_trace(raw,False)
        with self.assertRaises(AssertionError):norm_trace('GARNET_DECODE35_STAGE_WITNESS {}',False)
        self.assertEqual(norm_trace('',False)['full_attention_layers'],0)

    def test_both_defaultoff_profiles_require_complete_primary_and_53_coordinate_points(self):
        s,key,reports,reads,*_=sample()
        for phase in PHASES[:2]:
            report=reports[phase];old=copy.deepcopy(report);report['library_sha256']='b'*64
            q=control(report,old,reads[phase],reads[phase],'b'*64,'a'*64);self.assertEqual(q['primary']['complete_captures'],5);self.assertEqual(q['coordinate_timepoints'],53)

    def test_peer_row_last_component_or_unconsumed_ID_changes_reject_off_control(self):
        for mutation in ('peer','ID','library','trajectory','observer'):
            s,key,reports,reads,*_=sample();report=reports['off-candidate'];old=copy.deepcopy(report)
            if mutation=='peer':report['rows'][0]['post_step_states'][0]['capture']['logical']['components'][127]['row_sha256'][3]='f'*64
            elif mutation=='ID':report['rows'][0]['generated_ids'][3][52]+=1
            elif mutation=='library':report['library_sha256']='b'*64
            elif mutation=='trajectory':report['rows'][0]['key35_coordinate_trajectory']['points'][45]['repeat_carrier_hex'][0]='00000001'
            else:report['primary_outputs_only']=False
            with self.assertRaises(AssertionError):control(report,old,reads['off-candidate'],reads['off-candidate'],'a'*64,'a'*64)

    def test_exact_fixture_only_passes_selected_step52_never_full_admission(self):
        s,key,reports,reads,fr,prefix,files=sample();q=analyze(s,key,reports,reads,fr,prefix,lambda n:files[n],lambda n:files[n],'a'*64)
        self.assertTrue(q['selected_step52_intervention_passed']);self.assertFalse(q['full128_output_4K_MTP_quality_HTTP_or_performance_qualified']);self.assertFalse(q['two_times_target_achieved'])

    def test_earlier_consumed_token_change_prevents_same_input_cache_interpretation(self):
        s,key,reports,reads,fr,prefix,files=sample();reports['on-candidate']['rows'][0]['generated_ids'][0][30]+=1
        q=analyze(s,key,reports,reads,fr,prefix,lambda n:files[n],lambda n:files[n],'a'*64);self.assertFalse(q['all_four_consumed_52_IDs_exact']);self.assertNotIn('intervention',q);self.assertEqual(q['all_four_first_differing_ID_indices'][0],30)

    def test_only_last_unconsumed_prediction_change_stays_a_failed_intervention(self):
        s,key,reports,reads,fr,prefix,files=sample();reports['on-candidate']['rows'][0]['generated_ids'][2][52]+=1
        q=analyze(s,key,reports,reads,fr,prefix,lambda n:files[n],lambda n:files[n],'a'*64);self.assertTrue(q['all_four_consumed_52_IDs_exact']);self.assertFalse(q['selected_step52_intervention_passed'])

    def test_changed_prefill_graph_or_dll_cannot_be_attributed_to_norm(self):
        for field in ('graph_stats','library_sha256'):
            s,key,reports,reads,fr,prefix,files=sample();reports['on-candidate'][field]={} if field=='graph_stats' else 'b'*64
            with self.assertRaises(AssertionError):analyze(s,key,reports,reads,fr,prefix,lambda n:files[n],lambda n:files[n],'a'*64)

if __name__=='__main__':unittest.main()
