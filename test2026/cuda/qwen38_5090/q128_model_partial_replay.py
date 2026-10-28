"""A failed prefill's single stage observation is never a repeated admission."""
from pathlib import Path
import hashlib
import numpy as np
from .q128_model_layout import check_graphs
from .full_chunk_stage_layout import STAGES
from .full_chunk_stage_replay import frames,array as repeated_array
from .gdn_chunk_stage_replay import array as original_single_array
from .bf16_packed_state_gate import _validate,same_packed_bf16_row
from .kv128_prefix_cache_witness import same_state,snapshots
from .kv128_prefix_cache_replay import diagnose_state
sha=lambda b:hashlib.sha256(b).hexdigest()

def single_frame(report,read,off128):
    assert report['status']=='failed' and report['error']=='AssertionError()' and report['tokens']==128 and report['Q128_enabled'] is True and report['reference_profile']=='baseline' and report['stage_capture_enabled'] is True and not report['performance_admitted'] and report['original_reference_primary_equivalence_passed'] is False
    assert all(k not in report for k in ('ordinary_ids','ordinary_final','repeat_prefill','repeat_complete_prefill_state_vectors_exact','model_admission_passed'))
    check_graphs(report['graphs'],128,True,True);assert report['stage_runs']==31 and len(report['stage_frames'])==len(report['readonly_observations'])==1
    row=report['stage_frames'][0];assert row['start']==0 and row['count']==128 and set(row['arrays'])=={n for n,w in STAGES}
    for name,width in STAGES:assert row['arrays'][name]['words']==128*width;original_single_array(row['arrays'][name],read)
    witness=report['readonly_observations'][0];assert witness['start']==witness['run']==0 and witness['before']==witness['after']==off128['readonly_observations'][0]['before'];_validate(witness['before']['state']);assert witness['before']['state']['slots']==[0]
    c=witness['before']['controls'];assert set(c)=={'0','2','3','4'}
    for k,words in (('0',128*5120),('2',128),('3',384),('4',128)):assert c[k]['words']==words and len(c[k]['sha256'])==64
    return row

def replay(reports,reads):
    assert set(reports)==set(reads)=={'off32','off128','enabled128'};base=reports['off32'];candidate=reports['enabled128'];assert base['prompt_ids']==candidate['prompt_ids']==reports['off128']['prompt_ids'];aa=frames(base,reads['off32']);b=single_frame(candidate,reads['enabled128'],reports['off128'])
    same_state(candidate['empty_state'],base['empty_state'],reads['enabled128'],reads['off32']);assert not same_packed_bf16_row(candidate['ordinary_prefill']['logical'],0,base['ordinary_prefill']['logical'],0)
    state=diagnose_state(candidate['ordinary_prefill'],base['ordinary_prefill'],reads['enabled128'],reads['off32']);assert not state['physical_state_exact'] or not state['logical_state_exact']
    caches=snapshots(candidate,reads['enabled128'])==snapshots(base,reads['off32']);fixture=(Path(__file__).resolve().parent/'evidence/reconstructed-fa3-common-input.f32').read_bytes();assert sha(fixture)=='1f5f7340cac51fc1c56157fcf10a623d901de81e1456a74674b6408d5282c72d';rows={};first=None
    for name,width in STAGES:
        xraw=b''.join(repeated_array(a['arrays'][name],reads['off32']) for a in aa);yraw=original_single_array(b['arrays'][name],reads['enabled128']);assert len(xraw)==len(yraw)==128*width*4
        if name=='input':assert xraw==yraw==fixture
        x=np.frombuffer(xraw,dtype='<f4');y=np.frombuffer(yraw,dtype='<f4');changed=np.flatnonzero(x.view('<u4')!=y.view('<u4'));at=int(changed[0]) if changed.size else None
        row=dict(byte_exact=not bool(changed.size),changed_F32_words=int(changed.size),complete_F32_words_compared=128*width,maximum_absolute_difference=float(np.max(np.abs(x.astype(np.float64)-y.astype(np.float64)))),first_difference=None if at is None else dict(token=at//width,dimension=at%width,word=at,baseline=float(x[at]),candidate=float(y[at])));rows[name]=row
        if first is None and changed.size:first=dict(stage=name,**row['first_difference'])
    return dict(status='Q128_failed_original_prefill_single_first128_stages_diagnosed',original_full_model_prefill_gate_failed=True,complete_prefill_state=state,all12_layer3_prefix_cache_arrays_byte_exact=caches,complete_F32_words_compared=6553600,stages=rows,first_different_stage=first,candidate_complete_prefill_observations=1,candidate_repeat_or_decode_not_run=True,readonly_first128_graph_input_exact=True,causal_or_model_or_quality_or_performance_admission=False,new_benchmarks=False,two_times_target_achieved=False)
