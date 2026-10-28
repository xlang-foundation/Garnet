"""Separate failed-Q-only parity control and original correct-T32 Q+O admission."""
import hashlib,json,zipfile
from .o128_model_layout import reference,check_graphs
from .full_chunk_stage_layout import STAGES
from .full_chunk_stage_replay import frames,array
from .gdn_chunk_stage_replay import array as single_array
from .full_chunk_stage_vendor import stage_parity
from .kv128_prefix_cache_witness import same_state,snapshots
from .q128_model_partial_replay import single_frame
from .q128_model_replay import candidate_primary,differences
PARENT='q128-model-admission-attempt1';PARENT_SHA='5641e31a8b378f5fab423df84bfba38fccd9a0f184ad0bbd2d1c60f41828ab4a';sha=lambda b:hashlib.sha256(b).hexdigest()
def controls(reports,reads,proofs):
    assert set(reports)==set(reads)=={'off32','Q128_only_control'};archive=proofs/(PARENT+'-evidence.zip');assert sha(archive.read_bytes())==PARENT_SHA;result={}
    with zipfile.ZipFile(archive) as old:
        for key,phase,t,q in (('off32','off32',32,False),('Q128_only_control','enabled128',128,True)):
            r=reports[key];read=reads[key];base='results/'+PARENT+'/'+phase+'/';prior=json.loads(old.read(base+'report.json'));oldread=lambda n:old.read(base+n);admit=t==32
            assert r['status']=='full_chunk_stages_collected_primary_profile_exact' and r['tokens']==t and r['Q128_enabled'] is q and r['O128_enabled'] is False and (r['reference_job'],r['reference_profile'],r['reference_admits_model'])==reference(t,q,False) and r['reference_primary_equivalence_passed'] is True and r['original_reference_primary_equivalence_passed'] is admit and r['model_admission_passed'] is admit and r['instrumentation_equivalence_only'] is (not admit) and r['repeat_complete_prefill_state_vectors_exact'] is True
            check_graphs(r['graphs'],t,True,q,False)
            for name in ('empty_state','ordinary_prefill'):same_state(r[name],prior[name],read,oldread)
            same_state(r['repeat_prefill'],r['ordinary_prefill'],read,read);assert snapshots(r,read)==snapshots(prior,oldread)
            if t==32:
                assert r['ordinary_ids']==prior['ordinary_ids'] and len(r['ordinary_ids'])==128;same_state(r['ordinary_final'],prior['ordinary_final'],read,oldread);stages=stage_parity(r,prior,read,oldread)
            else:
                assert 'ordinary_ids' not in r and 'ordinary_final' not in r
                fresh=frames(r,read);saved=single_frame(prior,oldread,r);assert len(fresh)==1 and r['readonly_observations'][0]==prior['readonly_observations'][0]
                for name,width in STAGES:assert array(fresh[0]['arrays'][name],read)==single_array(saved['arrays'][name],oldread)
                stages=dict(complete_stage_entries=10,complete_F32_words=6553600,all_bytes_exact=True,fresh_two_complete_prefills_and_stage_repeats_exact=True,prior_candidate_single_observation_only=True,no_prior_repeat_or_decode_claim=True)
            result[key]=dict(complete_reference_states_vectors_all12cache_arrays_exact=True,stages=stages,model_admission=admit)
    return dict(status='O128_off_correctT32_and_failed_Q_only_controls_exact',controls=result,failed_Q_only_control_never_admission_oracle=True)
def replay(reports,reads,proofs):
    assert set(reports)==set(reads)=={'off32','Q128_only_control','Q128_plus_O128'};off=controls({k:reports[k] for k in ('off32','Q128_only_control')},{k:reads[k] for k in ('off32','Q128_only_control')},proofs);r=reports['Q128_plus_O128'];assert r['O128_enabled'] is True and r['Q128_enabled'] is True and r['reference_admits_model'] is True and r['reference_primary_equivalence_passed'] is True;check_graphs(r['graphs'],128,True,True,True)
    admitted=candidate_primary(r,reads['Q128_plus_O128'],proofs);stages=differences(reports['off32'],r,reads['off32'],reads['Q128_plus_O128']);stages['all_ten_stage_bytes_exact']=all(row['byte_exact'] for row in stages['stages'].values())
    return dict(status='O128_one_real3968plus128_text_case_original_correctT32_admission_replayed',defaultoff=off,one_case_admission=admitted,complete_first128_stages=stages,full_six_case_quality_or_performance_qualified=False,two_times_target_achieved=False)
