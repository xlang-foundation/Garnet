"""Full old primary/array replay plus new Z counter, no full-model cause claim."""
from .gdn_chunk_stage_replay import replay as previous_replay
from .fp8_z128_stage_layout import check_graph


def replay(report,read):
    for row in report['cases']:check_graph(row['graph'],row['tokens'],row['diagnostic'])
    result=previous_replay(report,read)
    exact=all(not row['different_stages'] and row['final_history_exact'] and row['final_state_exact'] for row in result['stage_comparisons'])
    return {'status':'fp8_z128_stage_primary_witness_arrays_replayed','array_replay':result,'native_Z128_counters_verified':True,'controlled_first128_all11_stages_exact':exact,'full3968_model_cause_established':False,'model_or_quality_or_performance_qualified':False,'two_times_target_achieved':False}
