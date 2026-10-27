"""Original complete GDN math gates plus exact arrays against the successful control."""
import hashlib,json,zipfile
from .gdn_chunk_stage_replay import replay as original_replay,array
from .gdn_kv128_stage_layout import check_graph
from .verify_gdn_chunk_stage_witness_bound import complete_entries
OLD='fp8-z128-native-package-audit-attempt1'
CONTROL_SHA='aad7fbece8b229c50a028106e4a241bd568556e025090982305730735bb341a6'
sha=lambda b:hashlib.sha256(b).hexdigest()
def compare_entries(now,prior,read,oldread):
    if isinstance(now,dict):
        assert isinstance(prior,dict)
        if set(now)=={'words','sha256','parts'}:
            assert set(prior)==set(now) and now['words']==prior['words']
            assert array(now,read)==array(prior,oldread),'Actual GDN control array changed'
            return 1
        return sum(compare_entries(v,prior[k],read,oldread) for k,v in now.items() if isinstance(v,(dict,list)) and k in prior)
    if isinstance(now,list):
        assert isinstance(prior,list) and len(now)==len(prior)
        return sum(compare_entries(a,b,read,oldread) for a,b in zip(now,prior))
    return 0
def replay(report,read,proofs):
    for row in report['cases']:check_graph(row['graph'],row['tokens'],row['diagnostic'])
    result=original_replay(report,read)
    assert result['complete_output_f32_words_replayed']==98205696
    exact=all(not row['different_stages'] and row['final_history_exact'] and row['final_state_exact'] for row in result['stage_comparisons'])
    entries=complete_entries(report,read);assert entries==509
    path=proofs/(OLD+'-evidence.zip');assert sha(path.read_bytes())==CONTROL_SHA
    with zipfile.ZipFile(path) as z:
        assert z.testzip() is None;prefix='results/'+OLD+'/native/'
        prior=json.loads(z.read(prefix+'report.json'))
        assert report['input_ids']==prior['input_ids'] and report['cases_sha256']==prior['cases_sha256'] and report['config_sha256']==prior['config_sha256'] and report['weight_metadata']==prior['weight_metadata']
        compared=compare_entries(report,prior,read,lambda n:z.read(prefix+n));assert compared==509
    return dict(status='actual_GDN50_original_stage_gates_and_all_control_arrays_replayed',original_array_replay=result,complete_finite_array_entries=entries,complete_control_array_entries_compared=compared,all_old_control_array_bytes_exact=True,controlled_first128_all11_stages_exact=exact,distinct_reconstructed_Z128_and_zero_KV128_routes_verified=True,full3968_model_or_quality_or_performance_qualified=False,two_times_target_achieved=False)
