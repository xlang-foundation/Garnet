"""Complete actual layer3 stage replay, strict primary equivalence before diagnosis."""
import hashlib,json,zipfile
import numpy as np
from .full_chunk_stage_layout import STAGES,check_graphs
from .bf16_packed_state_gate import _validate
from .layer_boundary_replay import primary_equivalence as original_primary_equivalence
from .gdn_chunk_stage_replay import array as original_array
sha=lambda b:hashlib.sha256(b).hexdigest()
CAPSULE='layer-boundary-attempt1-first128-capsule.zip';SHA_CAPSULE='292983c6b5301701db4f7b5763c4cd13cbd0254b46f865087490ea1e953557c7'
def primary(reports,reads,proofs):
    for mode in ('control','stages'):
        selected={phase:reports[mode+str(t)] for phase,t in (('baseline',32),('native',128))}
        for phase,r in selected.items():
            assert r['status']=='full_chunk_stages_collected_primary_profile_exact' and r['stage_capture_enabled'] is (mode=='stages')
            check_graphs(r['graphs'],r['tokens'],mode=='stages')
        # Explicit status-only metadata view reuses unchanged complete-state,
        # raw-vector, all12cache and correctT32 128ID admission predicates.
        view={phase:{**r,'status':'layer_boundaries_collected_primary_profile_exact'} for phase,r in selected.items()}
        result=original_primary_equivalence(view,{phase:reads[mode+str(t)] for phase,t in (('baseline',32),('native',128))},proofs)
    return dict(defaultoff_and_enabled_same_profile_complete_states_vectors_all12caches_exact=True,correct_T32_128IDs_and_finalstate_exact=True,failed_T128_only_instrumentation_equivalence_control=True)

def array(value,read):
    assert set(value)=={'words','parts','sha256','repeat_sha256'}
    raw=original_array({k:v for k,v in value.items() if k!='repeat_sha256'},read)
    assert value['sha256']==value['repeat_sha256'];return raw

def frames(report,read):
    t=report['tokens'];assert t in (32,128) and len(report['prompt_ids'])==3968 and report['stage_capture_enabled'] and not report['performance_admitted']
    check_graphs(report['graphs'],t,True);rows=report['stage_frames'];assert len(rows)==128//t
    for i,row in enumerate(rows):
        assert row['start']==i*t and row['count']==t and set(row['arrays'])=={n for n,w in STAGES}
        for n,w in STAGES:assert row['arrays'][n]['words']==t*w;array(row['arrays'][n],read)
    readonly=report['readonly_observations'];assert len(readonly)==2*len(rows) and report['stage_runs']==2*(3968//t)
    for repeat in (0,1):
        for i,row in enumerate(readonly[repeat*len(rows):(repeat+1)*len(rows)]):
            assert row['start']==i*t and row['run']==repeat*(3968//t)+i and row['before']==row['after']
            _validate(row['before']['state']);assert row['before']['state']['slots']==[i*t]
            controls=row['before']['controls'];assert set(controls)=={'0','2','3','4'}
            for key,words in (('0',t*5120),('2',t),('3',3*t),('4',t)):
                assert controls[key]['words']==words and len(controls[key]['sha256'])==64
            assert row['before']==readonly[i]['before']
    return rows

def replay(reports,reads,proofs):
    equivalence=primary(reports,reads,proofs);a=reports['stages32'];b=reports['stages128'];aa=frames(a,reads['stages32']);bb=frames(b,reads['stages128']);assert a['prompt_ids']==b['prompt_ids']
    from pathlib import Path
    common=Path(__file__).resolve().parent/'evidence/reconstructed-fa3-common-input.f32'
    rawinput=common.read_bytes();assert len(rawinput)==2621440 and sha(rawinput)=='1f5f7340cac51fc1c56157fcf10a623d901de81e1456a74674b6408d5282c72d'
    capsule=proofs/CAPSULE;assert sha(capsule.read_bytes())==SHA_CAPSULE
    with zipfile.ZipFile(capsule) as old:
        for key,rows,phase in (('stages32',aa,'baseline'),('stages128',bb,'native')):
            for row in rows:
                start=row['start'];t=row['count'];assert array(row['arrays']['input'],reads[key])==rawinput[start*5120*4:(start+t)*5120*4]
                assert array(row['arrays']['output'],reads[key])==old.read('results/layer-boundary-attempt1/'+phase+'/boundary-'+str(start)+'-L3.f32')
    result={};first=None
    for name,width in STAGES:
        raw_a=b''.join(array(row['arrays'][name],reads['stages32']) for row in aa);raw_b=array(bb[0]['arrays'][name],reads['stages128']);x=np.frombuffer(raw_a,dtype='<f4');y=np.frombuffer(raw_b,dtype='<f4');assert x.size==y.size==128*width
        changed=np.flatnonzero(x.view('<u4')!=y.view('<u4'));at=int(changed[0]) if changed.size else None
        row=dict(byte_exact=not changed.size,changed_F32_words=int(changed.size),complete_F32_words_compared=128*width,maximum_absolute_difference=float(np.max(np.abs(x.astype(np.float64)-y.astype(np.float64)))),first_difference=None if at is None else dict(token=at//width,dimension=at%width,word=at,baseline=float(x[at]),native=float(y[at])))
        row['byte_exact']=bool(row['byte_exact']);result[name]=row
        if first is None and changed.size:first=dict(stage=name,**row['first_difference'])
    return dict(status='actual_complete_first128_layer3_stages_replayed',primary_equivalence=equivalence,stages=result,first_different_stage=first,complete_F32_words_compared=128*sum(w for n,w in STAGES),complete_readonly_graph_input_state_and_controls=True,two_full_prefill_repeats_exact=True,stage_input_exact_retained_real_layer2_output=True,stage_output_exact_retained_same_profile_layer3_boundary=True,cause_established=False,model_or_performance_qualified=False,two_times_target_achieved=False)
