"""Complete finite boundary-array diagnosis; no model-admission tolerances."""
import hashlib,json,zipfile
import numpy as np
HIDDEN=5120
LAYERS=(0,1,2,3,4)
REFERENCE='kv128-prefix-cache-attempt1'
SHA_REFERENCE='12009807bcae0e3de5071010afbcba68f7b699f008c09c3656267d232b4a0ada'
def primary_equivalence(reports,reads,proofs):
    from .kv128_prefix_cache_witness import same_state,snapshots
    path=proofs/(REFERENCE+'-evidence.zip');assert hashlib.sha256(path.read_bytes()).hexdigest()==SHA_REFERENCE
    with zipfile.ZipFile(path) as z:
        assert z.testzip() is None
        for phase,tokens in (('baseline',32),('native',128)):
            r=reports[phase];base='results/'+REFERENCE+'/'+phase+'/';control=json.loads(z.read(base+'report.json'));oldread=lambda n:z.read(base+n)
            assert r['tokens']==tokens and r['reference_job']==REFERENCE and r['admission_oracle_job']=='mtp-prefill16-full4k-admission-attempt1' and r['prompt_ids']==control['prompt_ids']
            assert r['status']=='layer_boundaries_collected_primary_profile_exact' and r['model_admission_passed'] is (tokens==32)
            for name in ('empty_state','ordinary_prefill'):same_state(r[name],control[name],reads[phase],oldread)
            same_state(r['repeat_prefill'],r['ordinary_prefill'],reads[phase],reads[phase])
            assert snapshots(r,reads[phase])==snapshots(control,oldread)
            if tokens==32:
                assert r['ordinary_ids']==control['ordinary_ids'] and len(r['ordinary_ids'])==128
                same_state(r['ordinary_final'],control['ordinary_final'],reads[phase],oldread)
            else:assert 'ordinary_ids' not in r and 'ordinary_final' not in r
    return dict(same_profile_complete_primary_states_vectors_all12cache_arrays_exact=True,T32_retained128IDs_finalstate_exact=True,T128_known_failed_full_model_gate_preserved=True)

def frame(raw,row):
    assert len(raw)==row['words']*4 and hashlib.sha256(raw).hexdigest()==row['sha256']
    a=np.frombuffer(raw,dtype='<f4');assert np.isfinite(a).all();return a

def validate(report,read,expected_tokens=3968):
    tokens=report['tokens'];assert tokens in (32,128) and len(report['prompt_ids'])==expected_tokens
    assert report['same_profile_primary_equivalence_passed'] and not report['performance_admitted']
    rows=report['boundaries'];assert len(rows)==expected_tokens//tokens
    for index,row in enumerate(rows):
        assert row['start']==index*tokens and row['count']==tokens and set(row['layers'])==set(map(str,LAYERS))
        for value in row['layers'].values():
            assert value['words']==tokens*HIDDEN and value['repeat_sha256']==value['sha256'];frame(read(value['file']),value)
    assert report['boundary_runs']==2*len(rows) and report['repeat_complete_prefill_state_vectors_exact']
    return rows

def compare(base,native,read_base,read_native,expected_tokens=3968):
    a=validate(base,read_base,expected_tokens);b=validate(native,read_native,expected_tokens)
    assert base['tokens']==32 and native['tokens']==128 and base['prompt_ids']==native['prompt_ids']
    observations={};first=None
    for layer in LAYERS:
        changed=0;maximum=0.;initial=None
        for index,row in enumerate(b):
            target=row['layers'][str(layer)];y=frame(read_native(target['file']),target)
            controls=[a[4*index+i]['layers'][str(layer)] for i in range(4)]
            x=np.concatenate([frame(read_base(v['file']),v) for v in controls]);assert x.size==y.size==128*HIDDEN
            mask=x.view('<u4')!=y.view('<u4');count=int(np.count_nonzero(mask));changed+=count
            if count:
                maximum=max(maximum,float(np.max(np.abs(x.astype(np.float64)-y.astype(np.float64)))))
                if initial is None:
                    at=int(np.flatnonzero(mask)[0]);initial=dict(token=row['start']+at//HIDDEN,dimension=at%HIDDEN,frame_start=row['start'],word=at,baseline=float(x[at]),native=float(y[at]),baseline_files=[v['file'] for v in controls],native_file=target['file'])
        observations[str(layer)]=dict(byte_exact=changed==0,changed_F32_words=changed,complete_F32_words_compared=expected_tokens*HIDDEN,maximum_absolute_difference=maximum,first_difference=initial)
        if changed and first is None:first=dict(layer=layer,**initial)
    return dict(status='complete_layer0_to4_hidden_boundaries_diagnosed',layers=observations,first_different_layer=first,complete_F32_words_compared=expected_tokens*HIDDEN*len(LAYERS),all_boundaries_byte_exact=first is None,primary_same_profile_equivalence_only=True,original_correct_T32_model_admission_oracle_unchanged=True,cause_established=False,performance_admitted=False,two_times_target_achieved=False)
