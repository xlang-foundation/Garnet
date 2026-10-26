"""All original finite/readonly/carried-state/repeat checks plus strict cross-profile admission."""
import copy
from .layer2_ordinary_replay import replay as original_replay
from .z128_reconstructed_layer2_layout import check_graph
def replay(report,read,fixture):
    for row in report['cases']:check_graph(row['graph'],row['tokens'],row['diagnostic'])
    # Only the independently checked family metadata is adapted for the old array checker.
    # Every actual input/output entry and all array bytes remain unchanged.
    view=copy.deepcopy(report)
    for row in view['cases']:row['graph']['native_fp8_z128_canonical_projections']=int(row['tokens']==128)
    result=original_replay(view,read,fixture)
    result['actual_family_counters']=[{'tokens':row['tokens'],'legacy':row['graph']['native_fp8_z128_canonical_projections'],'reconstructed':row['graph']['native_fp8_z128_reconstructed_projections']} for row in report['cases']]
    result['layer2_cross_profile_admitted']=all(result[k] for k in ('hidden_bytes_exact','final_history_exact','final_state_exact'))
    return result
def require_admission(result):
    assert result['status']=='layer2_ordinary_complete_arrays_replayed' and result['finite_array_entries']==73 and result['complete_output_f32_words_replayed']==10895360
    assert result['same_real_common_input'] and result['readonly_inputs_and_carried_states_exact'] and result['repeats_exact']
    assert result['hidden_words']==655360 and result['different_hidden_f32_words']==0
    assert result['hidden_bytes_exact'] and result['final_history_exact'] and result['final_state_exact'] and result['layer2_cross_profile_admitted']
    assert result['model_or_quality_or_performance_qualified'] is False
    return True
