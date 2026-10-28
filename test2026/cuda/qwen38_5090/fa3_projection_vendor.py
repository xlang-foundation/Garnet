"""Strict byte-parity control for enabling only NVIDIA's cuBLASLt vendor logger."""
from .fa3_projection_witness import replay,STAGES
from .gdn_chunk_stage_replay import array
def compare(actual,prior,read,oldread,fixture):
    replay(actual,read,fixture);replay(prior,oldread,fixture)
    assert actual['weight_metadata']==prior['weight_metadata'] and actual['source_input_metadata']==prior['source_input_metadata']
    assert array(actual['common_input'],read)==array(prior['common_input'],oldread)==fixture;count=1
    for a,b in zip(actual['cases'],prior['cases']):
        assert a['tokens']==b['tokens'] and len(a['repeats'])==len(b['repeats'])==2
        for seq,oldseq in zip(a['repeats'],b['repeats']):
            assert len(seq)==len(oldseq)
            for current,old in zip(seq,oldseq):
                assert (current['start'],current['repeat'])==(old['start'],old['repeat'])
                for name in ('input_hidden','input_hidden_after'):
                    assert array(current[name],read)==array(old[name],oldread);count+=1
                for name,width in STAGES:
                    assert array(current['arrays'][name],read)==array(old['arrays'][name],oldread);count+=1
    assert count==51
    return dict(complete_finite_array_entries_compared=count,all_raw_bytes_exact=True,weight_metadata_and_real_input_metadata_exact=True,model_or_performance_qualified=False)
