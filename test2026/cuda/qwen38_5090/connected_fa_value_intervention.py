"""Exact connected-component off/on gate; no complete-model qualification."""
import numpy as np
from .connected_fa_value_gate import observe,admit as baseline_admit
from .decode_boundary_analysis import blob

def match_prior_baseline(report,prior,read,read_prior):
    assert report['mode']==prior['mode'] and report['physical_batch']==prior['physical_batch']
    batch=report['physical_batch'];assert batch in (1,4) and len(report['cases'])==len(prior['cases'])==(8 if batch==1 else 2)
    fields=('label','layer','actual_decode_step','physical_rows','input_cache_sha256_before','input_cache_sha256_after','input_after_sha256','weight_metadata')
    arrays=0
    for current,old in zip(report['cases'],prior['cases']):
        assert all(current[key]==old[key] for key in fields)
        assert blob(current['input'],read,batch*81920)==blob(old['input'],read_prior,batch*81920);arrays+=1
        assert len(current['repeats'])==len(old['repeats'])==2
        for i,(a,b) in enumerate(zip(current['repeats'],old['repeats'])):
            assert a['repeat']==b['repeat']==i and a['input_after_sha256']==b['input_after_sha256']
            for key,width in [('unmasked_normalized',81920),('component_hidden',81920),('component_V',4096)]:
                assert blob(a[key],read,batch*width)==blob(b[key],read_prior,batch*width),'New reference/off capture changed from paired prior component baseline';arrays+=1
    return dict(status='all_prior_component_input_normalizer_hidden_V_arrays_exact',arrays=arrays)

def admit(reference,off,on,read_reference,read_off,read_on,read_fixture):
    baseline=baseline_admit(reference,off,read_reference,read_off,read_fixture)
    intervention=observe(reference,on,read_reference,read_on,read_fixture)
    assert all(row['B4_component_vs_four_B1_different_words']==0 for row in intervention['rows']), 'V intervention differs from exact four-B1 active cache'
    def hidden_rows(report,read):
        values={};batch=report['physical_batch']
        for record in report['cases']:
            for repeat,entry in enumerate(record['repeats']):
                raw=blob(entry['component_hidden'],read,batch*81920)
                for position,row in enumerate(record['physical_rows']):
                    key=(record['label'],row,repeat);assert key not in values
                    values[key]=raw[position*81920:(position+1)*81920]
        assert len(values)==16
        return values
    expected=hidden_rows(reference,read_reference);old=hidden_rows(off,read_off);new=hidden_rows(on,read_on)
    assert set(expected)==set(old)==set(new)
    comparisons=[]
    for key in sorted(expected):
        x=np.frombuffer(expected[key],dtype='<u4')
        before=int(np.count_nonzero(x!=np.frombuffer(old[key],dtype='<u4')))
        after=int(np.count_nonzero(x!=np.frombuffer(new[key],dtype='<u4')))
        comparisons.append(dict(label=key[0],physical_row=key[1],repeat=key[2],full_T4_hidden_F32_words=20480,off_vs_four_B1_different_words=before,on_vs_four_B1_different_words=after))
    assert all(row['on_vs_four_B1_different_words']==0 for row in comparisons), 'Complete T4 component hidden arrays differ from four B1 rows'
    return dict(status='connected_FA_value_intervention_V_and_full_T4_component_hidden_exact',baseline=baseline,intervention=intervention,hidden=comparisons,original_failed_SDK_or_model_job_promoted=False,full_model_or_quality_or_performance_qualified=False,scope='Both actual FA3step5/FA11step1 preceding-input component cases; ordinary zero history/positions/slots; full input-cache/input-readonly/normalizer/repeat/finite gates inherited. Off must reproduce original B1/B4 caches; on must equal fourB1 activeV and complete T4 hidden arrays bitwise. Actual original full-model histories/state/output equivalence and performance require separate unchanged gates.')
