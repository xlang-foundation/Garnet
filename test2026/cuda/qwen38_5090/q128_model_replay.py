"""Strict original full-state/token admission, and complete first128 stage observations."""
from pathlib import Path
import hashlib,json,zipfile
import numpy as np
from .q128_model_layout import check_graphs,reference_profile
from .full_chunk_stage_layout import STAGES
from .full_chunk_stage_replay import frames,array
from .full_chunk_stage_vendor import stage_parity
from .layer_boundary_replay import primary_equivalence,REFERENCE,SHA_REFERENCE
from .kv128_prefix_cache_witness import same_state,snapshots
sha=lambda b:hashlib.sha256(b).hexdigest()
PARENT='full-chunk-stage-attempt2';PARENT_SHA='085ad330cde66fd755c383f05b7af16d610d7fec3464de6e4a2928a10da7a196'

def controls(reports,reads,proofs):
    assert set(reports)==set(reads)=={'off32','off128'};selected={};mapped={}
    for phase,key,t in (('baseline','off32',32),('native','off128',128)):
        r=reports[key];assert r['status']=='full_chunk_stages_collected_primary_profile_exact' and r['tokens']==t and r['Q128_enabled'] is False and r['reference_profile']==reference_profile(t,False) and r['original_reference_primary_equivalence_passed'];check_graphs(r['graphs'],t,True,False)
        selected[phase]={**r,'status':'layer_boundaries_collected_primary_profile_exact'};mapped[phase]=reads[key]
    primary=primary_equivalence(selected,mapped,proofs);archive=proofs/(PARENT+'-evidence.zip');assert sha(archive.read_bytes())==PARENT_SHA;parity={}
    with zipfile.ZipFile(archive) as old:
        for key,phase in (('off32','stages32'),('off128','stages128')):
            base='results/'+PARENT+'/'+phase+'/';prior=json.loads(old.read(base+'report.json'));parity[key]=stage_parity(reports[key],prior,reads[key],lambda n:old.read(base+n))
    return dict(status='Q128_defaultoff_complete_primary_and_stage_parity_passed',original_primary=primary,complete_stage_parity=parity,no_Q128_route_selected=True)

def candidate_primary(report,read,proofs):
    assert report['tokens']==128 and report['Q128_enabled'] is True and report['reference_profile']=='baseline' and report['status']=='full_chunk_stages_collected_primary_profile_exact' and report['model_admission_passed'] is True and report['original_reference_primary_equivalence_passed'] and report['instrumentation_equivalence_only'] is False and report['repeat_complete_prefill_state_vectors_exact'] is True and not report['performance_admitted'];check_graphs(report['graphs'],128,True,True)
    path=proofs/(REFERENCE+'-evidence.zip');assert sha(path.read_bytes())==SHA_REFERENCE
    with zipfile.ZipFile(path) as old:
        base='results/'+REFERENCE+'/baseline/';golden=json.loads(old.read(base+'report.json'));oldread=lambda n:old.read(base+n)
        assert report['reference_job']==REFERENCE and report['admission_oracle_job']=='mtp-prefill16-full4k-admission-attempt1' and report['prompt_ids']==golden['prompt_ids']
        for name in ('empty_state','ordinary_prefill','ordinary_final'):same_state(report[name],golden[name],read,oldread)
        same_state(report['repeat_prefill'],report['ordinary_prefill'],read,read)
        assert snapshots(report,read)==snapshots(golden,oldread)
        assert report['ordinary_ids']==golden['ordinary_ids'] and len(report['ordinary_ids'])==128
    return dict(status='Q128_one_real3968plus128_text_case_original_correctT32_admission_passed',all_complete_packed_states_raw_vectors_all12cache_arrays_128IDs_finalstate_exact=True,two_complete_prefill_repeats_exact=True,full_six_case_and_quality_performance_admission_still_required=True)

def differences(base,candidate,read_base,read_candidate):
    aa=frames(base,read_base);bb=frames(candidate,read_candidate);assert base['tokens']==32 and candidate['tokens']==128 and base['prompt_ids']==candidate['prompt_ids'];rows={};first=None
    common=Path(__file__).resolve().parent/'evidence/reconstructed-fa3-common-input.f32';fixture=common.read_bytes();assert sha(fixture)=='1f5f7340cac51fc1c56157fcf10a623d901de81e1456a74674b6408d5282c72d'
    for name,width in STAGES:
        a=b''.join(array(row['arrays'][name],read_base) for row in aa);b=array(bb[0]['arrays'][name],read_candidate);assert len(a)==len(b)==128*width*4
        if name=='input':assert a==b==fixture
        x=np.frombuffer(a,dtype='<f4');y=np.frombuffer(b,dtype='<f4');changed=np.flatnonzero(x.view('<u4')!=y.view('<u4'));at=int(changed[0]) if changed.size else None
        row=dict(byte_exact=bool(not changed.size),changed_F32_words=int(changed.size),complete_F32_words_compared=128*width,maximum_absolute_difference=float(np.max(np.abs(x.astype(np.float64)-y.astype(np.float64)))),first_difference=None if at is None else dict(token=at//width,dimension=at%width,word=at,baseline=float(x[at]),candidate=float(y[at])));rows[name]=row
        if first is None and changed.size:first=dict(stage=name,**row['first_difference'])
    return dict(status='complete_Q128_first128_all10_stage_difference_replayed',stages=rows,first_different_stage=first,complete_F32_words_compared=6553600,complete_readonly_graph_input_states_and_controls=True,all_complete_stage_parts_finite_and_repeats_exact=True,full_model_cause_not_assumed=True)

def replay(reports,reads,proofs):
    assert set(reports)==set(reads)=={'off32','off128','enabled128'}
    off=controls({k:reports[k] for k in ('off32','off128')},{k:reads[k] for k in ('off32','off128')},proofs);admission=candidate_primary(reports['enabled128'],reads['enabled128'],proofs);stages=differences(reports['off32'],reports['enabled128'],reads['off32'],reads['enabled128'])
    return dict(status='actual_Q128_defaultoff_parity_one_full_text_case_and_complete_stages_replayed',defaultoff=off,one_case_admission=admission,stages=stages,full_six_case_quality_or_performance_qualified=False,two_times_target_achieved=False)
