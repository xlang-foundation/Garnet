"""Original GDN route proof plus all sixteen FA V projections; no gate waiver."""
from .b4_b1_short_m16_gate import route_trace as gdn_route
from .v_m16_component_trace import route_trace as v_route

def route_trace(raw):
    gdn=gdn_route(raw);value=v_route(raw)
    assert value['initializations_by_output_rows']['1024']>=16, 'Complete 64-layer decode requires all sixteen FA V projections'
    return dict(status='actual_B4_GDN_and_V_M16_full_model_routes_replayed',GDN=gdn,V=value,model_or_quality_or_performance_qualified=False)
