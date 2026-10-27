"""Physical Q/O/KV/reconstructed-Z routes, followed by the unchanged six-case oracle."""
from .kv128_prefix_cache_witness import check_graphs as physical_KV,OLD,RECON,KV
from .fp8_z128_full4k_model import validate_targets as legacy_targets,validate_selection as legacy_selection,replay_full4k_model as legacy_replay
Q='native_fp8_q128_canonical_projections';O='native_fp8_o128_canonical_projections';STAGE='native_full_chunk_stage_witnesses'
def count(row,field,expected):
    assert isinstance(row,dict) and type(row.get(field)) is int and row[field]==expected,(field,row.get(field),expected)
    return row[field]
def logical_targets(graphs,policy):
    assert type(policy) is str and policy=='1';physical_KV(graphs,128);view={};counts={}
    for chunk,row in graphs.items():
        expected=16 if chunk=='128' else 0
        counts[chunk]={Q:count(row,Q,expected),O:count(row,O,expected),STAGE:count(row,STAGE,0)};counts[chunk].update({field:row[field] for field in (OLD,RECON,KV,'native_fp8_m128_canonical_projections')});view[chunk]=dict(row)
        # Metadata-only legacy logical-Z view, after strict physical 0/48 family
        # checks. Raw stats and all numerical/ID/state/vector evidence stay intact.
        view[chunk][OLD]=row[OLD]+row[RECON]
    legacy_targets(view,policy);return view,counts
def validate_targets(graphs,policy):
    _,counts=logical_targets(graphs,policy);return dict(physical_Q_O_stage_counts=counts,physical_GDN48_reconstructedZ48_KV32_verified=True)
def selection_view(report):
    for name in ('fp8_m128_policy','fp8_z128_policy','fp8_q128_policy','fp8_o128_policy'):assert type(report[name]) is str and report[name]=='1'
    view,counts=logical_targets(report['fp8_target_graphs'],report['fp8_m128_policy']);aux=report['fp8_auxiliary_graphs'];assert set(aux)=={'draft4','draft16','prefill_norm','vision'};aux_counts={}
    for name,row in aux.items():
        aux_counts[name]=None if row is None and name=='prefill_norm' else {field:count(row,field,0) for field in (Q,O,KV,OLD,RECON,STAGE)}
    logical={**report,'fp8_target_graphs':view};legacy=legacy_selection(logical)
    return logical,dict(physical_target_counts=counts,physical_auxiliary_counts=aux_counts,legacy_logical_selection=legacy,raw_report_unchanged=True)
def validate_selection(report):return selection_view(report)[1]
def replay_full4k_model(report,read,reference,read_reference,cases):
    logical,selection=selection_view(report);result=legacy_replay(logical,read,reference,read_reference,cases)
    return {**result,'physical_Q_O_KV_reconstructed_Z_selection':selection,'all_original_six_case_state_vector_ID_callback_reset_predicates_unchanged':True}
