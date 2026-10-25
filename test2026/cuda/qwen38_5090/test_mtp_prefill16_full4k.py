"""Synthetic byte/state fault controls only; never GPU or model results."""
import copy,hashlib,struct,unittest
from .mtp_prefill16_full4k import ORDER,replay_report
from .test_bf16_packed_state_gate import witness


def fixture():
    files={};cases=[{'key':'synthetic-'+str(i),'case':'text' if i<2 else 'image','request_sha256':hashlib.sha256(str(i).encode()).hexdigest(),'image_sha256':None if i<2 else str(i)*64,'prompt_ids':[i+10]*3968,'generated_ids_by_mode':{'ordinary':[i+20]*128,'mtp':[i+20]*128}} for i in range(4)]
    def state(label,slots,seed):
        logical=witness(1);logical['slots']=[slots]
        for name in ('physical_row_sha256','logical_row_sha256'):logical[name]=[hashlib.sha256((name+str(seed)).encode()).hexdigest()]
        for i,c in enumerate(logical['components']):c['row_sha256']=[hashlib.sha256((str(seed)+':'+str(i)).encode()).hexdigest()]
        vectors={}
        for kind in ('hidden','normalized'):
            raw=struct.pack('<f',float(seed))*5120;name=label+'-'+kind+'.f32';files[name]=raw;vectors[kind]={'file':name,'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}
        return {'logical':logical,'vectors':vectors}
    report={'status':'mtp_prefill16_full4k_collected_needs_replay','physical_batch':1,'capacity':4096,'quality_qualified':False,'performance_admitted':False,'empty_state':state('empty',0,0),'final_reset':state('final-reset',0,0),'cases':[]}
    for sequence,index in enumerate(ORDER):
        c=cases[index];stem=str(sequence);ids=c['generated_ids_by_mode']['ordinary'];report['cases'].append({'sequence':sequence,'case_index':index,'key':c['key'],'request_sha256':c['request_sha256'],'image_sha256':c['image_sha256'],'prompt_ids':c['prompt_ids'],'ordinary_ids':ids[:],'candidate_ids':ids[:],'reset_state':state(stem+'-reset',0,0),'ordinary_prefill':state(stem+'-op',3968,index+1),'candidate_prefill':state(stem+'-cp',3968,index+1),'ordinary_final':state(stem+'-of',4095,index+11),'candidate_final':state(stem+'-cf',4095,index+11),'prefill_first_tokens':[[ids[0]]],'callbacks':[[0,t,i+1] for i,t in enumerate(ids)],'sampled_tokens_including_stop':[128],'stop_token_ids':[None],'speculative_metrics':{'mtp_prefill_calls':249}})
    return report,files,cases


class Full4KControls(unittest.TestCase):
    def test_all_six_synthetic_cases_and_32_complete_states_replay(self):
        report,files,cases=fixture();result=replay_report(report,files.__getitem__,cases);self.assertEqual(result['captured_states'],32);self.assertEqual([r['reset_repeat'] for r in result['cases']],[False]*4+[True]*2);self.assertFalse(result['performance_admitted'])
    def test_counts_callback_prefix_depth_geometry_and_early_stop_cannot_change(self):
        for kind in ('prompt','output','calls','callback','first','slot','stop','batch','missing','order'):
            report,files,cases=fixture();record=report['cases'][-1]
            if kind=='prompt':record['prompt_ids']=record['prompt_ids'][:-1]
            elif kind=='output':record['candidate_ids'].pop()
            elif kind=='calls':record['speculative_metrics']['mtp_prefill_calls']=993
            elif kind=='callback':record['callbacks'][-1][2]=127
            elif kind=='first':record['prefill_first_tokens']=[]
            elif kind=='slot':record['candidate_final']['logical']['slots']=[4096]
            elif kind=='stop':record['stop_token_ids']=[2]
            elif kind=='batch':report['physical_batch']=True
            elif kind=='missing':report['cases'].pop()
            else:record['case_index']=1
            with self.subTest(kind=kind),self.assertRaises((AssertionError,ValueError)):replay_report(report,files.__getitem__,cases)
    def test_full_component_and_final_reset_hash_difference_rejected(self):
        for name in ('candidate_prefill','candidate_final','reset_state','final_reset'):
            report,files,cases=fixture();state=report[name] if name=='final_reset' else report['cases'][-1][name];state['logical']['components'][-1]['row_sha256']=['f'*64]
            with self.subTest(name=name),self.assertRaises(AssertionError):replay_report(report,files.__getitem__,cases)
    def test_finite_vector_difference_with_correct_file_hash_is_rejected(self):
        report,files,cases=fixture();entry=report['cases'][-1]['candidate_final']['vectors']['normalized'];raw=struct.pack('<f',99.)*5120;files[entry['file']]=raw;entry['sha256']=hashlib.sha256(raw).hexdigest()
        with self.assertRaises(AssertionError):replay_report(report,files.__getitem__,cases)
    def test_nonfinite_and_truncated_vectors_rejected(self):
        for raw in (struct.pack('<f',float('nan'))*5120,b'\x00'*20476):
            report,files,cases=fixture();entry=report['empty_state']['vectors']['hidden'];files[entry['file']]=raw;entry['sha256']=hashlib.sha256(raw).hexdigest()
            with self.assertRaises(AssertionError):replay_report(report,files.__getitem__,cases)
    def test_reset_repeat_cannot_drift_even_when_ordinary_and_candidate_agree(self):
        report,files,cases=fixture()
        for name in ('ordinary_final','candidate_final'):report['cases'][-1][name]['logical']['physical_row_sha256']=['f'*64]
        with self.assertRaises(AssertionError):replay_report(report,files.__getitem__,cases)


if __name__=='__main__':unittest.main()
